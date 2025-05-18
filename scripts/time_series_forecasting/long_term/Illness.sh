export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=CAPTime
llm_ckp_dir=openai-community/gpt2
gpu_id=2
bs=32  
llm_hidden_size=768  
test_dir="1"
des=base
task=long_term_forecast
epochs=16


# training
python -u run.py \
  --use_wandb False\
  --patience 3\
  --des $des \
  --task_name $task \
  --is_training 1 \
  --root_path dataset/illness \
  --data_path illness.csv \
  --model_id illness_36 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 36 \
  --label_len 18 \
  --token_len 18 \
  --test_seq_len 36 \
  --test_label_len 18 \
  --test_pred_len 24 \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 512 \
  --train_epochs 16 \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir

# testing
for test_pred_len in 24 36 48 60
do
python -u run.py \
  --des $des \
  --task_name $task \
  --is_training 0 \
  --root_path dataset/illness \
  --data_path illness.csv \
  --model_id illness_36 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 36 \
  --label_len 18 \
  --token_len 18 \
  --test_seq_len 36 \
  --test_label_len 18 \
  --test_pred_len $test_pred_len \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 512 \
  --train_epochs $epochs \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir
done
