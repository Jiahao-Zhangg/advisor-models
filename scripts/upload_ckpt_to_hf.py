"""Export a SkyRL FSDP checkpoint to HuggingFace format and optionally upload to the Hub.

This script is meant to be run with torchrun, matching the checkpoint's world size.

Example:
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 advisor-models/scripts/upload_ckpt_to_hf.py \\
    --global_step_dir advisor-models/ckpts/mtob/global_step_12 \\
    --out_dir advisor-models/exports/mtob/global_step_12/policy \\
    --repo_id YOUR_USERNAME/mtob-advisor-step12
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import HfApi

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKYRL_TRAIN_ROOT = _REPO_ROOT / "SkyRL" / "skyrl-train"
sys.path.insert(0, str(_SKYRL_TRAIN_ROOT))

from skyrl_train.distributed.fsdp_strategy import FSDPStrategy  # noqa: E402
from skyrl_train.distributed.fsdp_utils import (  # noqa: E402
    get_init_weight_context_manager,
)


def _get_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def _init_distributed() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exporting this SkyRL FSDP checkpoint (expects nccl + GPUs).")

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return local_rank

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def _load_trainer_cfg(global_step_dir: Path):
    trainer_state_path = global_step_dir / "trainer_state.pt"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer state: {trainer_state_path}")
    obj = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    if "config" not in obj:
        raise ValueError(f"trainer_state.pt missing 'config': {trainer_state_path}")
    return obj["config"]


def _export_hf(
    *,
    global_step_dir: Path,
    policy_ckpt_dir: Path,
    out_dir: Path,
) -> None:
    cfg = _load_trainer_cfg(global_step_dir)

    fsdp_config = OmegaConf.create(dict(cfg.trainer.policy.fsdp_config))
    fsdp_strategy = str(cfg.trainer.strategy)

    strategy = FSDPStrategy(
        fsdp_config=fsdp_config,
        optimizer_config=None,
        fsdp_strategy=fsdp_strategy,
        seed=int(cfg.trainer.seed),
        micro_train_batch_size_per_gpu=1,
        train_batch_size=dist.get_world_size(),
        num_training_steps=None,
    )
    strategy.setup_distributed()

    hf_cfg_dir = policy_ckpt_dir / "huggingface"
    if not hf_cfg_dir.exists():
        raise FileNotFoundError(f"Missing HF config/tokenizer dir: {hf_cfg_dir}")

    model_config = AutoConfig.from_pretrained(hf_cfg_dir, trust_remote_code=True)
    init_ctx = get_init_weight_context_manager(
        use_meta_tensor=True,
        mesh=strategy.device_mesh,
    )

    with init_ctx():
        model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

    model = strategy.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(hf_cfg_dir, trust_remote_code=True)

    _, _states = strategy.load_ckpt(  # noqa: F841
        model=model,
        ckpt_dir=str(policy_ckpt_dir),
        optimizer=None,
        scheduler=None,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_only=True,
    )

    out_dir = out_dir.resolve()
    if dist.get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[rank-0] Exporting HuggingFace model to: {out_dir}")

    strategy.save_hf_model(model, str(out_dir), tokenizer=tokenizer)
    dist.barrier()


def _upload_folder(*, folder_path: Path, repo_id: str, token: str, private: bool) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        commit_message=f"Upload exported SkyRL checkpoint from {folder_path.name}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SkyRL FSDP checkpoint to HuggingFace format and upload to the Hub."
    )
    parser.add_argument(
        "--global_step_dir",
        type=str,
        required=True,
        help="Path like advisor-models/ckpts/mtob/global_step_12",
    )
    parser.add_argument(
        "--policy_subdir",
        type=str,
        default="policy",
        help="Subdirectory under global_step_dir containing the policy checkpoint (default: policy).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for HuggingFace model files (rank 0 writes).",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace repo id like 'username/model-name'. If omitted, no upload is performed.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token. If omitted, uses HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_HUB_TOKEN.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private repo.",
    )
    args = parser.parse_args()

    _init_distributed()

    global_step_dir = Path(args.global_step_dir).expanduser().resolve()
    policy_ckpt_dir = global_step_dir / args.policy_subdir
    out_dir = Path(args.out_dir).expanduser()

    if not policy_ckpt_dir.exists():
        raise FileNotFoundError(f"Policy checkpoint dir not found: {policy_ckpt_dir}")

    _export_hf(global_step_dir=global_step_dir, policy_ckpt_dir=policy_ckpt_dir, out_dir=out_dir)

    if args.repo_id:
        token = _get_token(args.token)
        if not token:
            raise ValueError(
                "Missing HuggingFace token. Provide --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            )
        if dist.get_rank() == 0:
            print(f"[rank-0] Uploading {out_dir} to HuggingFace Hub repo: {args.repo_id}")
            _upload_folder(folder_path=out_dir, repo_id=args.repo_id, token=token, private=args.private)
        dist.barrier()

    if dist.get_rank() == 0:
        print("[rank-0] Done.")


if __name__ == "__main__":
    main()
