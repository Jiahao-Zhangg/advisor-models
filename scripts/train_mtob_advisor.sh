#!/bin/bash

# Training script for MTOB domain

# Resolve repo root (directory containing this scripts/ folder).
ADVISOR_MODELS_ROOT="${ADVISOR_MODELS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ADVISOR_MODELS_ROOT" || exit 1

# Set environment variables
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="$ADVISOR_MODELS_ROOT/SkyRL/skyrl-train:$PYTHONPATH"
export DATA_DIR="$ADVISOR_MODELS_ROOT/data/mtob"
# Note: if you set CUDA_VISIBLE_DEVICES=2,3, GPUs will be re-indexed as 0,1 inside the process.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export NUM_GPUS=2
export LOGGER="wandb"  # change to "console" to print to stdout

mkdir -p "$ADVISOR_MODELS_ROOT/ckpts/mtob"

# Run training
"$ADVISOR_MODELS_ROOT/SkyRL/skyrl-train/.venv/bin/python" -m advisor_models.mtob.main_mtob \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=100 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=2048\
  generator.sampling_params.max_generate_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=mtob \
  generator.n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="advisor_models" \
  trainer.run_name="mtob" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$ADVISOR_MODELS_ROOT/ckpts/mtob"
