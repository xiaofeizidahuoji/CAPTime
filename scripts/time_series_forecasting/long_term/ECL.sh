export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=3600

model_name=CAPTime
llm_ckp_dir=openai-community/gpt2
gpu_id=2
bs=2048  
llm_hidden_size=768  
test_dir=long
des=baseL
task=long_term_forecast
epochs=15
lr=0.003

# training one model with a context length
torchrun --nnodes 1 --nproc-per-node 2 run.py \
  --use_wandb True\
  --patience 3\
  --des $des \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len 96 \
  --batch_size $bs \
  --learning_rate $lr \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --train_epochs $epochs \
  --more_experts \
  --use_amp \
  --use_multi_gpu \
  --gpu $gpu_id \
  --tmax $epochs \
  --cosine \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir \

# testing the model on all forecast lengths
for test_pred_len in 720
do
python run.py \
  --task_name long_term_forecast \
  --des $des \
  --is_training 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_512 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len $test_pred_len \
  --batch_size $bs \
  --learning_rate $lr \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --train_epochs $epochs \
  --more_experts \
  --use_amp \
  --llm_ckp_dir $llm_ckp_dir \
  --gpu $gpu_id \
  --mix_embeds \
  --tmax $epochs \
  --cosine
done