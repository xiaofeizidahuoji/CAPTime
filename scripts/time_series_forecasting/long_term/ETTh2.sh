export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=CAPTime
llm_ckp_dir=openai-community/gpt2

gpu_id=0
bs=256  
llm_hidden_size=768  
test_dir="1"
des=base
task=long_term_forecast
epochs=12


# training one model with a context length
python -u run.py \
  --use_wandb False\
  --patience 3\
  --des $des \
  --task_name $task \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len 96 \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 256 \
  --train_epochs $epochs \
  --use_amp \
  --use_mlp \
  --more_mse \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir \

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --des $des \
  --task_name $task \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_512 \
  --model $model_name \
  --data ETTh2 \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len $test_pred_len \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 256 \
  --train_epochs $epochs \
  --use_amp \
  --use_mlp \
  --gpu $gpu_id \
  --cosine \
  --tmax 10 \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir
done
