import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import GPT2Tokenizer
from layers.mlp import MLP
from layers.layers import *
from transformers import GPT2Config, GPT2Model
from models.TTM.modeling_tinytimemixer import TinyTimeMixerForPrediction, TinyTimeMixerConfig, TinyTimeMixerBlock
from scipy.stats import kurtosis, skew

def get_des_dict():
    des_dict = {}
    des_dict['Weather'] = 'This is a meteorological dataset recorded at 10-minute intervals across the entire year of 2020.'
    des_dict['ETTh'] = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
    des_dict['ETTm'] = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
    des_dict['ECL'] = 'Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.This archive contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months).'
    des_dict['illness'] = 'The Illness dataset captures weekly patient counts and influenza-like illness rates.'

    return des_dict


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.token_len = configs.token_len
        self.seq_len = configs.seq_len
        self.pred_len = configs.token_len
        self.num_token = self.seq_len//self.token_len
        self.des_dict = get_des_dict()
        self.task = configs.task_name
        dataset = '_'.join(configs.model_id.split('_')[:-1])
        print(dataset)
        if 'ETTh' in dataset:
            dataset = 'ETTh'
        elif 'ETTm' in dataset:
            dataset = 'ETTm'
        self.des = self.des_dict[dataset] if dataset in self.des_dict else ' '
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        ### ttm
        self.ttm_hidden = 192
        ttm_dir = 'ibm-granite/granite-timeseries-ttm-r1'
        original_config = TinyTimeMixerConfig.from_pretrained(ttm_dir)
        self.ts_encoder = TinyTimeMixerForPrediction.from_pretrained(ttm_dir).backbone.encoder
        total_params = sum(p.numel() for p in self.ts_encoder.parameters())


        ### pretrained gpt2
        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim_of_gpt2 = 768
        self.recon_cri = nn.MSELoss()

        ### sampling
        self.num_samples = 5
        self.strategy = 'max'
        self.distribution = 'studentt'
        self.temperature = 0.5

        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        for name, param in self.ts_encoder.named_parameters():
            param.requires_grad = False
        ## new patcher/mixer
        if self.token_len != 64:
            self.ts_encoder.patcher = nn.Linear(self.token_len, self.ttm_hidden)
            for name, param in self.ts_encoder.patcher.named_parameters():
                param.requires_grad = True
        if self.num_token != 8:
            original_config.num_patches = self.num_token
            self.ts_encoder.mlp_mixer_encoder = TinyTimeMixerBlock(config=original_config)
            for name, param in self.ts_encoder.mlp_mixer_encoder.named_parameters():
                param.requires_grad = True

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.adapter = nn.Linear(self.ttm_hidden, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.adapter = MLP(self.ttm_hidden, self.hidden_dim_of_gpt2, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            if configs.use_mlp:
                self.mlp = MLP(self.hidden_dim_of_gpt2, self.hidden_dim_of_gpt2, 
                                configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                                configs.dropout, configs.mlp_activation)
            self.cross = CrossAttention_qformer(self.hidden_dim_of_gpt2, num_attention_heads=8, encoder_hidden_size=self.hidden_dim_of_gpt2,
                                        attention_probs_dropout_prob=configs.dropout, num_query_token=self.num_token)
            if configs.more_experts:
                k,n = 4,8
                if dataset=='Weather':
                    k,n = 2,8
            else:
                k,n = 2,4
            self.decoder = Moe_distribution(self.hidden_dim_of_gpt2, configs.mlp_hidden_layers,
                                            configs.mlp_activation, self.token_len, dist_type=self.distribution,
                                            topk=k, N=n)
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text=None):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        bs, _, n_vars = x_enc.shape
        
        x_enc = x_enc.permute(0, 2, 1)
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]

        ### TTM encoder
        comit_loss = 0
        patch_embed = self.ts_encoder.patcher(fold_out)
        times_embeds_ttm, all_hidden = self.ts_encoder.mlp_mixer_encoder(patch_embed.unsqueeze(1))
        times_embeds = self.adapter(times_embeds_ttm.squeeze())
        
        ### text abstraction
        if text is None:
            prompt_emb = self.get_prompt_input(x_enc.unsqueeze(-1))
        else:
            prompt_emb = self.get_news_prompt(x_enc.unsqueeze(-1), text)
        if self.configs.use_mlp:
            text_prompt_emb = self.cross(kv=self.mlp(prompt_emb))[0]
        else:
            text_prompt_emb = self.cross(kv=prompt_emb)[0]

        times_embeds = times_embeds+text_prompt_emb
        outputs = self.gpt2(
            inputs_embeds=times_embeds).last_hidden_state

        ### context-aware distribution modeling
        time_tokens = outputs[:,-token_num:]
        if self.training:
            dec_out, balance_loss = self.decoder(time_tokens, text_prompt_emb)
            targets = (x_dec - means)/stdev
            if self.distribution == 'studentt':
                if self.task=='short_term_forecast':
                    loss = self.weighted_studentt_nll_loss(dec_out, targets.permute(0,2,1))
                else:
                    loss = self.studentt_nll_loss(dec_out, targets.permute(0,2,1))
                loc = dec_out[...,1].permute(0,2,1)
                loc = loc * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
                loc = loc + \
                    (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
                return loss+balance_loss, loc
        else: # infer
            dec_out = self.decoder(time_tokens, text_prompt_emb)
            if self.distribution == 'studentt':
                df = dec_out[..., 0]
                loc = dec_out[..., 1]
                scale = dec_out[..., 2]
                dist = torch.distributions.StudentT(df, loc, scale)
            if self.strategy == "max":
                out = loc
            out = out.permute(0, 2, 1)
            out = out * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
            out = out + \
                (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
            return out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, text=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, text)

    def studentt_nll_loss(self, dist_params, targets, scale_penalty=0.1, df_penalty=0.1):
        loc = dist_params[..., 1]

        df = dist_params[..., 0].clamp(min=1.1)
        scale = dist_params[..., 2].clamp(min=1e-6)
        z = (targets - loc) / scale
        log_prob = (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(df * math.pi * scale**2)
            - 0.5 * (df + 1) * torch.log1p(z**2 / df)
        )
        # nll loss
        mse_loss = F.mse_loss(loc, targets)
        nll_loss = -log_prob.mean()
        if self.configs.more_mse:
            return 0.8*nll_loss + 0.2*mse_loss
        # elif 'text' in self.configs.task:
        #     scale_reg = scale_penalty * torch.mean(scale) 
        #     df_reg = df_penalty * torch.mean(df)        
        #     return nll_loss + scale_reg
        else:
            return nll_loss
        


    def get_prompt_input(self,x_enc):
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        trends = x_enc.diff(dim=1).sum(dim=1)
        stds = torch.std(x_enc, dim=1)
        q1 = torch.quantile(x_enc, q=0.25, dim=1)
        q3 = torch.quantile(x_enc, q=0.75, dim=1)
        diffs = x_enc.diff(dim=1)
        max_slopes = diffs.abs().max(dim=1).values
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.des}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
            )

            prompt.append(prompt_)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
        
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x_enc.device))
        return prompt_embeddings

    

    def get_news_prompt(self,x_enc,text_info):
        prompt = []
        for b in range(x_enc.shape[0]):
            prompt_ = (
                f"<|start_prompt|>Make predictions about the future based on the following information: {text_info[b]};"
            )
            prompt.append(prompt_)
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
        prompt_embeddings = self.gpt2.get_input_embeddings()(prompt.to(x_enc.device))
        return prompt_embeddings