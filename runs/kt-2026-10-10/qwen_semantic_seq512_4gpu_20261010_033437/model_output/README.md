---
library_name: transformers
license: other
base_model: /mnt/data2/models/Qwen3.5-122B-A10B
tags:
- llama-factory
- full
- kt-kernel
- generated_from_trainer
model-index:
- name: model_output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# model_output

This model is a fine-tuned version of [/mnt/data2/models/Qwen3.5-122B-A10B](https://huggingface.co//mnt/data2/models/Qwen3.5-122B-A10B) on the fft_real_100 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 4
- total_eval_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- training_steps: 15

### Training results



### Framework versions

- Transformers 5.6.0
- Pytorch 2.9.1+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2
