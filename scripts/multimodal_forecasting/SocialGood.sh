export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=CAPTime
llm_ckp_dir=openai-community/gpt2
gpu_id=0
bs=32 
llm_hidden_size=768  
test_dir="1"
des=base
task=with_text_forecast
epochs=12


# training
python -u run.py \
  --use_wandb False\
  --patience 5\
  --des $des \
  --task_name $task \
  --is_training 1 \
  --root_path dataset/data_with_text/SocialGood \
  --data_path Unadj_UnemploymentRate_ALL_processed.csv \
  --model_id SocialGood_8 \
  --model $model_name \
  --data custom_text \
  --seq_len 8 \
  --label_len 7 \
  --token_len 1 \
  --test_seq_len 8 \
  --test_label_len 7 \
  --test_pred_len 6 \
  --batch_size $bs \
  --learning_rate 0.0001 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 256 \
  --train_epochs $epochs \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir \

# testing
for test_pred_len in 6 8 10 12
do
python -u run.py \
  --des $des \
  --task_name $task \
  --is_training 0 \
  --root_path dataset/data_with_text/SocialGood \
  --data_path Unadj_UnemploymentRate_ALL_processed.csv \
  --model_id SocialGood_8 \
  --model $model_name \
  --data custom_text \
  --seq_len 8 \
  --label_len 7 \
  --token_len 1 \
  --test_seq_len 8 \
  --test_label_len 7 \
  --test_pred_len $test_pred_len \
  --batch_size $bs \
  --learning_rate 0.0001 \
  --mlp_hidden_layers 1 \
  --llm_hidden_size $llm_hidden_size \
  --mlp_hidden_dim 256 \
  --train_epochs $epochs \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax 10 \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir
done
