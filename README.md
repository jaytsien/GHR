<h3 align="center">
<p>GHR
<a href="https://github.com/jaytsien/GHR/blob/main/LICENSE">
   <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
</a>
</h3>
<div align="center">
    <p>Capturing Conversational Interaction for Question Answering via <b>G</b>lobal <b>H</b>istory <b>R</b>easoning
    <p>NAACL Findings 2022
</div>

<div align="center">
  <img alt="GHR Overview" src="https://github.com/jaytsien/GHR/blob/main/utils/GHR_model.png" width="800px">
</div> 	


We present GHR for conversational question answering (CQA). You can train ELECTRA by using our framework, GHR, described in our [paper](https://aclanthology.org/2022.findings-naacl.159.pdf). 


## Requirements

```bash
$ conda create -n GHR python=3.8.10
$ conda activate GHR
$ conda install tqdm
$ conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
$ pip install transformers==3.5.0
```

### Datasets

We use the [QuAC (Choi et al., 2018)](https://quac.ai/) dataset for training and evaluating our models, and test on the leaderboard.

## Train

The following example fine-tunes ELECTRA on the QuAC dataset by using GHR. 
We performed all experiments using a single 16GB GPU (Tesla V100).

```bash
INPUT_DIR=./datasets/
OUTPUT_DIR=./tmp/model

CUDA_VISIBLE_DEVICES=0 python3 run_quac.py \
	--model_type electra  \
	--model_name_or_path   electra-large \
	--do_train \
	--do_eval \
        --data_dir ${INPUT_DIR} \
	--train_file train.json \
	--predict_file dev.json \
	--output_dir ${OUTPUT_DIR} \
	--per_gpu_train_batch_size 12 \
	--num_train_epochs 2 \
	--learning_rate 2e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--do_lower_case \
	--fp16 --fp16_opt_level "O2" \
	--evaluate_during_training \
	--max_answer_length 50 --cache_prefix electra-large
```

By default, we use mixed precision apex `--fp16` for acceleration training and prediction. 

## Evaluation

The following example evaluates our trained model with the development set of QuAC.

```bash
INPUT_DIR=./datasets/
MODEL_DIR=./tmp/model/
OUTPUT_DIR=./tmp/

CUDA_VISIBLE_DEVICES=0 python3 run_quac.py \
	--model_type electra  \
	--model_name_or_path   ${MODEL_DIR} \
	--do_eval \
        --data_dir ${INPUT_DIR} \
	--train_file train.json \
	--predict_file dev.json \
	--output_dir ${OUTPUT_DIR} \
	--per_gpu_train_batch_size 12 \
	--num_train_epochs 2 \
	--learning_rate 2e-5 \
	--weight_decay 0.01 \
	--threads 20 \
	--do_lower_case \
	--fp16 --fp16_opt_level "O2" \
	--evaluate_during_training \
	--max_answer_length 50 --cache_prefix electra-large
```

### Result

Evaluating models trained with predefined hyperparameters yields the following results:

```bash
DEV Results: {'F1': 74.9}  TEST Results: {'F1': 73.7}
```

## Citation

```bibtex
@inproceedings{qian2022capturing,
  title={Capturing Conversational Interaction for Question Answering via Global History Reasoning},
  author={Qian, Jin and Zou, Bowei and Dong, Mengxing and Li, Xiao and Aw, Aiti and Hong, Yu},
  booktitle={Findings of the Association for Computational Linguistics: NAACL 2022},
  pages={2071--2078},
  year={2022}
}
```
