import torch
import torch.nn as nn
import math
from typing import Optional, List
import torch.nn.functional as F
from layers.mlp import MLP

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)


class CrossAttention_qformer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, encoder_hidden_size, attention_probs_dropout_prob=0.0, num_query_token=8):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=0.02)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [batch, num_heads, seq_length, head_size]

    def forward(
        self,
        kv: torch.Tensor,        # Encoder output (key and value)
        kv_attention_mask: Optional[torch.Tensor] = None,  # Mask for encoder
    ) -> torch.Tensor:
        # Compute query, key, and value projections
        query_layer = self.transpose_for_scores(self.query(self.query_tokens))
        key_layer = self.transpose_for_scores(self.key(kv))
        value_layer = self.transpose_for_scores(self.value(kv))
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [batch, num_heads, query_len, key_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply encoder attention mask
        if kv_attention_mask is not None:
            attention_scores = attention_scores + kv_attention_mask
        # Normalize to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch, num_heads, query_len, head_size]
        # Reshape context layer back to [batch, query_len, hidden_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs


## distri_decoder
class DistributionHead(nn.Module):
    def __init__(self, hidden_dim, token_len, layers, act, dist_type='studentt'):
        super().__init__()
        self.dist_type = dist_type
        self.token_len = token_len
        dim_dict = {'studentt':3, 'gaussian':2, 'laplace':2}
        self.output_dim = dim_dict[dist_type]
        self.param_proj =  MLP(hidden_dim, self.output_dim*token_len,
                                hidden_dim//2, layers,
                                0.1, act)
        
    def forward(self, x):
        B = x.shape[0]
        params = self.param_proj(x) 
        if self.dist_type == 'studentt':
            df, loc, scale = params.chunk(3, dim=-1)
            df = F.softplus(df) + 1.0   
            scale = F.softplus(scale)   
            return torch.stack([df, loc, scale], dim=-1).reshape(B,1,-1,self.output_dim)  # [..., pred_len, 3]
        elif self.dist_type == 'gaussian':
            loc, scale = params.chunk(2, dim=-1)
            scale = F.softplus(scale)  
            return torch.stack([loc, scale], dim=-1).reshape(B,1,-1,self.output_dim)  # [..., pred_len, 2]
        elif self.dist_type == 'laplace':
            loc, scale = params.chunk(2, dim=-1)
            return torch.stack([loc, scale.exp()], dim=-1).reshape(B,1,-1,self.output_dim)

class Moe_distribution(nn.Module):
    def __init__(self, hidden_dim, mlp_hidden_layers, mlp_activation, token_len, dist_type='studentt', topk=3, N=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_len = token_len
        self.topk = topk
        self.num_experts = N
        self.balance_lambda = 0.1
        dim_dict = {'studentt':3, 'gaussian':2, 'laplace':2}
        self.output_dim = dim_dict[dist_type]

        self.gate = nn.Linear(hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            DistributionHead(hidden_dim, self.token_len, mlp_hidden_layers,
                             act=mlp_activation, dist_type=dist_type) for _ in range(self.num_experts)
        ])
    

    def forward(self, x, text=None):
        batch_size, num_token, dim = x.shape
        x = x.reshape(-1, dim)
        if text is not None:
            text = text.reshape(-1, dim)
            logits = self.gate(text)
        else:
            logits = self.gate(x)

        # Routing and softmax
        routing_weights, selected_experts = torch.topk(logits, self.topk, dim=-1)
        routing_weights = routing_weights.to(x.dtype)  # [B*num_token, topk]
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)
        balance_loss = self.compute_balance_loss(logits, selected_experts)

        # Expert forward
        final_hidden_states = torch.zeros(
            (batch_size*num_token, self.token_len, self.output_dim), dtype=x.dtype, device=x.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if len(top_x) == 0: 
                continue
            current_state = x[top_x]
            current_hidden_states = expert_layer(current_state).reshape(-1,self.token_len,self.output_dim) * routing_weights[top_x, idx, None, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        if self.training:
            return final_hidden_states.reshape(batch_size,1,-1,self.output_dim), balance_loss
        else:
            return final_hidden_states.reshape(batch_size,1,-1,self.output_dim)
    
    def compute_balance_loss(self, gate_logits, selected_experts):
        importance = torch.sum(F.softmax(gate_logits, dim=-1), dim=0)  # [num_experts]
        mask = F.one_hot(selected_experts, num_classes=self.num_experts)  # [B*T, topk, N]
        load = torch.sum(mask.float(), dim=(0,1))  # [num_experts]
        importance_loss = torch.std(importance) / (torch.mean(importance) + 1e-6)
        load_loss = torch.std(load) / (torch.mean(load) + 1e-6)
        return self.balance_lambda * (importance_loss + load_loss)