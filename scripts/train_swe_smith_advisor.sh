#!/bin/bash

# Training script for SWE-Smith domain

# Set environment variables
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export PYTHONPATH="/advisor-models/SkyRL/skyrl-train:$PYTHONPATH"
export DATA_DIR="/advisor-models/data/swe_smith"
export NUM_GPUS=8
export LOGGER="wandb"  # change to "console" to print to stdout
export ADVISOR_MODELS_MODE="advisor"  # "advisor" or "baseline"
export AGENT_MODEL="gemini/gemini-2.5-flash"  # Student model for advisor mode

# IMPORTANT: Update these URLs to point to your agent and eval servers
export AGENT_SERVER_URL="http://YOUR_VM_IP:8081"  # CHANGE THIS to your VM's IP (agent server)
export EVAL_SERVER_URL="http://YOUR_VM_IP:8082"   # CHANGE THIS to your VM's IP (eval server)

# Repository to train on (e.g., jd__tenacity)
export REPO="jd__tenacity"

# Run training
/advisor-models/SkyRL/skyrl-train/.venv/bin/python -m advisor_models.swe_smith.main_swe_smith \
    data.train_data="['$DATA_DIR/train_${REPO}.json']" \
    data.val_data="['$DATA_DIR/validation_${REPO}.json']" \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.policy.model.path="Qwen/Qwen2.5-7B-Instruct" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp2 \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
    trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
    generator.num_inference_engines=$NUM_GPUS \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.epochs=40 \
    trainer.eval_batch_size=49 \
    trainer.eval_before_train=true \
    trainer.eval_interval=10 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=8 \
    trainer.policy_mini_batch_size=4 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.ckpt_interval=20 \
    trainer.max_prompt_length=28672 \
    generator.sampling_params.max_generate_length=4096 \
    generator.sampling_params.top_p=0.999 \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.algorithm.use_kl_loss=true \
    generator.backend=vllm \
    generator.run_engines_locally=true \
    generator.weight_sync_backend=nccl \
    generator.async_engine=true \
    generator.batched=false \
    environment.env_class=swe_smith \
    generator.n_samples_per_prompt=8 \
    generator.gpu_memory_utilization=0.8 \
    trainer.logger="$LOGGER" \
    trainer.project_name="advisor_models" \
    trainer.run_name="swe_smith_${REPO}" \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/swe_smith_${REPO}" \
    generator.zero_reward_on_non_stop=true \
    trainer.hf_save_interval=20 \
    trainer.export_path="$HOME/exports/swe_smith_${REPO}"
