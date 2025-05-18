import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_with_text_forecasting import Exp_With_Text_Forecast


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise ValueError("Input must be 'true' or 'false'")

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)

    parser = argparse.ArgumentParser(description='CAPTime')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='CAPTime',
                        help='')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--test_data_path', type=str, default='ETTh1.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')
    
    parser.add_argument('--use_wandb', type=str, default='False',
                        help='use wandb')  # can be base

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--percent', type=int, default=100, help='percentage of using training data')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default='models--meta-llama--Llama-2-7b-hf', help='llm checkpoints dir')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')
    parser.add_argument('--llm_hidden_size', type=int, default=4096, help='hidden size of llm')
    parser.add_argument('--more_experts', action='store_true', help='', default=False)
    parser.add_argument('--use_mlp', action='store_true', help='', default=False)
    parser.add_argument('--more_mse', action='store_true', help='', default=False)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    args = parser.parse_args()

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "4"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    
    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'with_text_forecast':
        Exp = Exp_With_Text_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_lr{}_bt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.learning_rate,
                args.batch_size,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_lr{}_bt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.learning_rate,
            args.batch_size,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
