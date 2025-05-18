export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=CAPTime
llm_ckp_dir=openai-community/gpt2
gpu_id=1
llm_hidden_size=768
test_dir="1"
des=base
task=short_term_forecast 

python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly_512 \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des $des \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --more_experts \
  --use_mlp \
  --cosine \
  --train_epochs 12 \
  --tmax 12 \
  --llm_ckp_dir $llm_ckp_dir \
  --weight_decay 0.000005


python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly_512 \
  --model $model_name \
  --data m4 \
  --batch_size 32 \
  --des $des \
  --itr 1 \
  --learning_rate 0.001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --more_experts \
  --use_mlp \
  --cosine \
  --train_epochs 14 \
  --tmax 14 \
  --llm_ckp_dir $llm_ckp_dir \
  --weight_decay 0.00001

python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_512 \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --des $des \
  --itr 1 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --more_experts \
  --use_mlp \
  --cosine \
  --train_epochs 15 \
  --tmax 15 \
  --llm_ckp_dir $llm_ckp_dir \
  --weight_decay 0.000001

python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly_512 \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des $des \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'SMAPE' \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --more_experts \
  --use_mlp \
  --cosine \
  --llm_ckp_dir $llm_ckp_dir \
  --train_epochs 12 \
  --tmax 12 \
  --weight_decay 0.00001

python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily_512 \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des $des \
  --itr 1 \
  --learning_rate 0.0005 \
  --loss 'SMAPE' \
  --use_amp \
  --use_mlp \
  --mlp_hidden_dim 1024 \
  --more_experts \
  --llm_ckp_dir $llm_ckp_dir \
  --weight_decay 0.00001

python -u run.py \
  --gpu $gpu_id \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly_512 \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des $des \
  --itr 1 \
  --learning_rate 0.0003 \
  --loss 'SMAPE' \
  --use_amp \
  --use_mlp \
  --mlp_hidden_dim 1024 \
  --more_experts \
  --weight_decay 0.00001 \
  --cosine \
  --llm_ckp_dir $llm_ckp_dir \
  --train_epochs 12 \
  --tmax 12 \