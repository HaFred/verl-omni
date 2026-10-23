# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Allowlisted Co-RL optimizer observations, collected after init and restore."""

import hashlib
import json
import math
import os
import re
from pathlib import Path


def policy_parameter_digest(module) -> dict:
    """Hash every local actor parameter without gathering FSDP shards or changing state.

    Compare each rank against itself across calibration phases. This includes frozen
    backbone and adapter parameters, but excludes gradients, buffers and optimizer
    state; it is not a standalone receipt of what a serving engine loaded.
    """
    import torch
    from torch.distributed.tensor import DTensor

    checksum = hashlib.sha256()
    count = local_numel = trainable_local_numel = 0
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        if local.device.type == "meta" or local.layout != torch.strided:
            raise ValueError("Policy hashing requires materialized dense local parameters")
        value = local.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite policy parameter: {name}")
        layout = {
            "name": name,
            "dtype": str(value.dtype),
            "global_shape": list(parameter.shape),
            "local_shape": list(value.shape),
            "requires_grad": parameter.requires_grad,
            "placements": [str(item) for item in parameter.placements] if isinstance(parameter, DTensor) else [],
        }
        header = json.dumps(layout, sort_keys=True).encode()
        checksum.update(len(header).to_bytes(8, "big"))
        checksum.update(header)
        # uint8 supports bf16 without a dtype conversion; memoryview avoids copying
        # large parameter byte strings in addition to any required CPU transfer.
        raw = value.reshape(-1).view(torch.uint8).numpy()
        checksum.update(memoryview(raw))
        count += 1
        local_numel += value.numel()
        if parameter.requires_grad:
            trainable_local_numel += value.numel()
    if not count:
        raise ValueError("Policy hashing requires actor parameters")
    return {
        "parameters_sha256": checksum.hexdigest(),
        "parameter_count": count,
        "local_numel": local_numel,
        "trainable_local_numel": trainable_local_numel,
        "scope": "rank_local_parameters_only",
    }


def optimizer_snapshot(engine, *, phase: str, checkpoint_path=None) -> dict:
    """Read live optimizer groups; never substitute configured peak learning rates."""
    if phase not in {"initialized", "restored"}:
        raise ValueError("Unknown optimizer provenance phase")
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None:
        raise ValueError("Co-RL optimizer is unavailable for runtime provenance")
    lanes = {"und_lora": "und", "gen_lora": "gen", "other_trainable": "other"}
    lrs = {"und": None, "gen": None}
    groups = []
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in lanes:
            raise ValueError("Unclassified Co-RL optimizer group")
        lr = float(group["lr"])
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("Optimizer learning rate must be finite and nonnegative")
        lane = lanes[name]
        if any(item["lane"] == lane for item in groups):
            raise ValueError("Duplicate Co-RL optimizer group")
        groups.append({"lane": lane, "lr": lr, "parameter_count": len(group["params"])})
        if lane in lrs:
            lrs[lane] = lr
    if not groups:
        raise ValueError("Co-RL optimizer has no groups")
    step = re.search(r"(?:^|[/\\])global_step_(\d+)(?:[/\\]|$)", str(checkpoint_path or ""))
    return {
        "phase": phase,
        "checkpoint_step": int(step.group(1)) if step else None,
        "optimizer_group_lrs": lrs,
        "groups": groups,
    }


def write_runtime_provenance(output_dir, worker_snapshots: list[list[dict]], *, trainer_step: int) -> Path:
    """Persist all workers' observations and promote the manifest to actual LRs.

    Only the trainer writes the manifest. Rank order is the worker-group RPC order;
    all ranks must agree on init/restore sequence, group identity and current LR.
    """
    if not worker_snapshots or any(not history for history in worker_snapshots):
        raise ValueError("Missing runtime optimizer provenance from actor worker")
    reference = worker_snapshots[0]
    for history in worker_snapshots[1:]:
        if history != reference:
            raise ValueError("Actor workers disagree on runtime optimizer provenance")
    event = {"trainer_step": int(trainer_step), "workers": worker_snapshots}
    output = Path(output_dir) / "run_manifest.json"
    manifest = json.loads(output.read_text(encoding="utf-8"))
    manifest.setdefault("runtime_optimizer_snapshots", []).append(event)
    manifest["optimizer_group_lrs"] = reference[-1]["optimizer_group_lrs"]
    manifest["optimizer_lrs_source"] = "runtime"
    # The existing allowlisted config and its source hashes remain launch evidence.
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, output)
    return output
