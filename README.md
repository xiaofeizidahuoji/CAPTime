# CAPTime
The implementation of "Context-Aware Probabilistic Modeling with LLM for
Multimodal Time Series Forecasting".
## Setup
### Environment
Install Pytorch and necessary dependencies.
```bash
pip install -r requirements.txt
```

### Dataset
Download the dataset from [Google Drive](https://drive.google.com/file/d/16MvSnIi6hX3G7kiR_el-FQbYCdSuw80T/view?usp=sharing). Move the downloaded dataset to the `dataset/` directory. Extract the dataset to ensure the files are organized under the `dataset/` directory.

## Training Instructions
All scripts of tasks are put under ```scripts/```. To train the model, modify the parameters in the script. The framework will automatically perform training and testing.
### Multimodal forecasting
```bash
bash scripts/multimodal_forecasting/Agriculture.sh
```
### Short-term forecatsing
```bash
bash scripts/time_series_forecasting/short_term/M4.sh
```
### Long-term forecasting
```bash
bash scripts/time_series_forecasting/long_term/ETTh1.sh
```


## Parameters Setting
### Basic Parameters

```bash
model_name=CAPTime                 # Name of the model 
gpu_id=2                           # GPU ID to use
bs=2048                            # Batch size
llm_hidden_size=768                # Hidden size of the LLM
des=base                          # Description tag for the run
task=long_term_forecast            # Task type
epochs=15                          # Number of training epochs
lr=0.003                           # Learning rate
```

### Training Parameters
The main training script is executed using `torchrun` with the following parameters:

```bash
--patience 3                      # Early stopping patience
--seq_len 512                     # Input sequence length
--label_len 448                   # Label length
--token_len 64                    # Token length
--weight_decay 0.00001            # Weight decay for optimizer
--mlp_hidden_dim 1024             # Hidden dimension for MLP
--more_experts                    # Flag to use more experts
--use_multi_gpu                   # Enable multi-GPU training
```

### Testing Parameters
The testing script runs with similar parameters but with different prediction lengths:
```bash
--test_seq_len 512                # Test sequence length
--test_label_len 448              # Test label length
--test_pred_len $test_pred_len    # Varies per test
```

## Acknowledgement
Our work is built upon the code base of [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [MM-TSFlib](https://github.com/AdityaLab/MM-TSFlib), which we have adapted to meet the requirements of our research. We sincerely appreciate the authors for making their implementations and associated resources publicly available.
