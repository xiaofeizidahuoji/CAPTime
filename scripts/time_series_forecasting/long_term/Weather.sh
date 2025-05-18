export CUDA_VISIBLE_DEVICES=0,1,2,3
model_name=CAPTime
llm_ckp_dir=openai-community/gpt2
gpu_id=0
bs=512  
llm_hidden_size=768  
test_dir="long_term_forecast_ETTh1_512_64_VLM4TS_Qwen2vl_ETTh1_sl512_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl0_cosTrue_mixTrue_try2_nfft96_centerTrue_phase_instance_resize_position_drivinggpt_llm_mask_0"
des=base
epochs=15


# training one model with a context length
python -u run.py \
  --use_wandb False\
  --patience 3\
  --des $des \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_512 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len 96 \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --weight_decay 0.000005 \
  --mlp_hidden_layers 1 \
  --mlp_hidden_dim 512 \
  --llm_hidden_size $llm_hidden_size \
  --train_epochs $epochs \
  --more_experts \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --des $des \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id Weather_512 \
  --model $model_name \
  --data custom_no_time \
  --seq_len 512 \
  --label_len 448 \
  --token_len 64 \
  --test_seq_len 512 \
  --test_label_len 448 \
  --test_pred_len $test_pred_len \
  --batch_size $bs \
  --learning_rate 0.0005 \
  --weight_decay 0.000005 \
  --mlp_hidden_layers 1 \
  --mlp_hidden_dim 512 \
  --llm_hidden_size $llm_hidden_size \
  --train_epochs $epochs \
  --more_experts \
  --use_amp \
  --gpu $gpu_id \
  --cosine \
  --tmax $epochs \
  --drop_last \
  --llm_ckp_dir $llm_ckp_dir \
  --test_dir $test_dir
done
