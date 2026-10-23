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

"""Opt-in receipts for merged, unquantized BAGEL GEN publications.

These receipts verify transport completion and two selected row-parallel
sentinels. They do not establish full serving-model identity or scoring parity.
No source tensor is retained beyond its normal streaming lifetime.
"""

import copy
import hashlib
import json
import os
import socket

import torch

GEN_SENTINELS = {
    "transformer.layers.0.self_attn.o_proj_moe_gen.weight": "model.layers.0.self_attn.o_proj.gen_exp.weight",
    "transformer.layers.1.mlp_moe_gen.down_proj.weight": "model.layers.1.mlp_moe_gen.down_proj.weight",
}
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _dtype(value):
    name = str(value).removeprefix("torch.")
    name = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(name, name)
    if name not in _DTYPES:
        raise ValueError(f"Unsupported GEN receipt dtype: {name}")
    return _DTYPES[name]


def _digest(tensor):
    tensor = tensor.detach().contiguous().cpu()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("GEN sentinel has non-finite values")
    descriptor = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return {**descriptor, "sha256": digest.hexdigest()}


class _ExportSummary:
    def __init__(self, tp_size, dtype):
        if type(tp_size) is not int or tp_size < 1:
            raise ValueError("GEN receipt requires a positive TP size")
        self.dtype = _dtype(dtype)
        self.names = set()
        self.summary = {
            "tp_size": tp_size,
            "dtype": str(self.dtype),
            "tensor_count": 0,
            "byte_count": 0,
            "sentinels": {},
        }

    def observe(self, name, tensor):
        if name in self.names:
            raise ValueError(f"Duplicate exported GEN tensor: {name}")
        self.names.add(name)
        self.summary["tensor_count"] += 1
        self.summary["byte_count"] += tensor.numel() * tensor.element_size()
        if name in GEN_SENTINELS:
            size = self.summary["tp_size"]
            if tensor.ndim != 2 or tensor.shape[1] % size:
                raise ValueError("GEN sentinel does not support equal row-parallel shards")
            width = tensor.shape[1] // size
            self.summary["sentinels"][name] = [
                _digest(tensor[:, rank * width : (rank + 1) * width].to(self.dtype)) for rank in range(size)
            ]

    def finish(self):
        if set(self.summary["sentinels"]) != set(GEN_SENTINELS):
            raise ValueError("GEN export missing required sentinels")
        return copy.deepcopy(self.summary)


def summarize_export(weights, *, tp_size, dtype):
    observer = _ExportSummary(tp_size, dtype)
    for name, tensor in weights:
        observer.observe(name, tensor)
    return observer.finish()


def native_sentinels(pipeline, *, include_digests=True):
    """Read actual native tensors and their loader TP metadata, failing closed.

    Installed BAGEL aliases pipeline.language_model and bagel.language_model.
    Its MoTRowParallelLinear.gen_exp uses the parent's RowParallelLinear loader;
    both selected native weights are unquantized and split along input_dim=1.
    """
    language = getattr(getattr(pipeline, "bagel", None), "language_model", None)
    if language is None or language is not getattr(pipeline, "language_model", None):
        raise ValueError("GEN receipt requires the native BAGEL language-model alias")
    if (
        getattr(pipeline.bagel, "_sp_size", 1) != 1
        or getattr(getattr(pipeline.bagel, "parallel_config", None), "cfg_parallel_size", 1) != 1
    ):
        raise ValueError("GEN receipts do not support native SP or CFG parallelism")
    parameters = dict(language.named_parameters())
    result = {"sentinels": {}}
    for actor_name, native_name in GEN_SENTINELS.items():
        parameter = parameters.get(native_name)
        if parameter is None:
            raise ValueError(f"Missing native GEN sentinel: {native_name}")
        module = language.get_submodule(native_name.rsplit(".", 1)[0])
        if type(getattr(module, "quant_method", None)).__name__ != "UnquantizedLinearMethod":
            raise ValueError("GEN receipts require unquantized native row-parallel weights")
        rank, size = getattr(module, "tp_rank", None), getattr(module, "tp_size", None)
        if (
            type(rank) is not int
            or type(size) is not int
            or size < 1
            or not 0 <= rank < size
            or parameter.ndim != 2
            or getattr(parameter, "input_dim", None) != 1
            or getattr(parameter, "is_sharded_weight", False)
            or getattr(module, "input_size_per_partition", None) != parameter.shape[1]
            or getattr(module, "input_size", None) != parameter.shape[1] * size
        ):
            raise ValueError("Unsupported native GEN row-parallel layout")
        _dtype(parameter.dtype)
        if "tp_rank" in result and (result["tp_rank"], result["tp_size"]) != (rank, size):
            raise ValueError("Native GEN sentinels disagree on TP topology")
        result.update(tp_rank=rank, tp_size=size)
        evidence = (
            _digest(parameter) if include_digests else {"dtype": str(parameter.dtype), "shape": list(parameter.shape)}
        )
        result["sentinels"][actor_name] = {"native_name": native_name, **evidence}
    return result


def verify_sentinels(expected, observed):
    """Join actor-export expectations and observed native TP-local tensors."""
    rank, size = observed["tp_rank"], observed["tp_size"]
    if size != expected["tp_size"] or type(rank) is not int or not 0 <= rank < size:
        raise ValueError("GEN receipt TP topology mismatch")
    if set(observed["sentinels"]) != set(GEN_SENTINELS):
        raise ValueError("Missing observed GEN sentinel")
    for name in GEN_SENTINELS:
        native = observed["sentinels"][name]
        actor = expected["sentinels"][name][rank]
        if native.get("native_name") != GEN_SENTINELS[name] or any(
            native.get(key) != value for key, value in actor.items()
        ):
            raise ValueError(f"GEN sentinel mismatch on TP rank {rank}: {name}")


def validate_native_capabilities(results, *, expected_tp_size, dtype):
    """Require every native worker's layout preflight before any IPC send starts."""
    if type(expected_tp_size) is not int or expected_tp_size < 1:
        raise ValueError("Invalid expected native GEN TP topology")
    expected_dtype = str(_dtype(dtype))
    records = []

    def visit(value):
        if isinstance(value, list | tuple):
            for item in value:
                visit(item)
        elif isinstance(value, dict) and set(value.get("sentinels", {})) == set(GEN_SENTINELS):
            records.append(value)
        else:
            raise ValueError("Missing native GEN capability response")

    visit(results)
    if (
        len(records) != expected_tp_size
        or {r["tp_rank"] for r in records} != set(range(expected_tp_size))
        or any(r["tp_size"] != expected_tp_size for r in records)
    ):
        raise ValueError("Incomplete native GEN capability TP coverage")
    for record in records:
        for actor_name, native_name in GEN_SENTINELS.items():
            evidence = record["sentinels"][actor_name]
            if (
                evidence.get("native_name") != native_name
                or evidence.get("dtype") != expected_dtype
                or len(evidence.get("shape", [])) != 2
                or any(type(n) is not int or n <= 0 for n in evidence["shape"])
            ):
                raise ValueError("Unsupported native GEN capability dtype/name/shape")
            if evidence["shape"] != records[0]["sentinels"][actor_name]["shape"]:
                raise ValueError("Unequal native GEN TP shard shapes")
    return records


def prepare_adapter_capabilities(adapter, results):
    """Install an explicit complete native preflight gate on this sender rank."""
    adapter._bagel_corl_gen_receipt_capabilities = None
    records = validate_native_capabilities(
        results, expected_tp_size=adapter.config.tensor_model_parallel_size, dtype=adapter.config.dtype
    )
    channels = [r.get("base_channel_sha256") for r in records]
    own_channel = hashlib.sha256(adapter.zmq_handle.encode()).hexdigest()
    if (
        any(not isinstance(channel, str) or len(channel) != 64 for channel in channels)
        or len(set(channels)) != len(channels)
        or own_channel not in channels
    ):
        raise ValueError("GEN preflight missing unique matching native transport channels")
    adapter._bagel_corl_gen_receipt_capabilities = copy.deepcopy(records)
    return {"base_channel_sha256": own_channel, "tp_size": adapter.config.tensor_model_parallel_size}


async def send_receipted_weights(adapter, weights, *, global_steps, sender_cls, make_id, make_handle, **kwargs):
    """Stock merged IPC ordering with an update-scoped handle and actual stream audit."""
    adapter._bagel_corl_gen_receipt = None
    if kwargs.get("peft_config") is not None or kwargs.get("delta_flush"):
        raise ValueError("GEN receipts support merged full weights only")
    for field in ("ulysses_degree", "ring_degree", "pipeline_model_parallel_size", "cfg_parallel_size"):
        if getattr(adapter.config, field, 1) != 1:
            raise ValueError(f"GEN receipts do not support {field} parallelism")
    capabilities = getattr(adapter, "_bagel_corl_gen_receipt_capabilities", None)
    if capabilities is None:
        raise ValueError("GEN receipt transport requires complete native preflight before any receiver starts")
    prepare_adapter_capabilities(adapter, capabilities)
    observer = _ExportSummary(adapter.config.tensor_model_parallel_size, adapter.config.dtype)
    sequence = getattr(adapter, "_bagel_corl_gen_receipt_seq", 0)
    adapter._bagel_corl_gen_receipt_seq = sequence + 1
    update_id = make_id(global_steps, sequence)

    def stream():
        for name, tensor in weights:
            observer.observe(name, tensor)
            yield name, tensor
        observer.finish()

    future = await adapter._execute_method(
        "update_weights_from_ipc",
        non_block=True,
        kwargs={**kwargs, "use_shm": adapter.use_shm, "zmq_update_id": update_id, "bagel_corl_gen_receipt": True},
    )
    handle = make_handle(adapter.zmq_handle, update_id)
    sender = sender_cls(
        zmq_handle=handle,
        bucket_size_mb=adapter.config.checkpoint_engine.update_weights_bucket_megabytes,
        use_shm=adapter.use_shm,
    )
    await sender.async_send_weights(stream())
    if future is not None:
        await future
    if adapter._has_server:
        await adapter.server_handle.clear_kv_cache.remote()
        if global_steps is not None:
            await adapter.server_handle.set_global_steps.remote(global_steps)
    receipt = {
        **observer.finish(),
        "update_id": update_id,
        "terminal_acknowledged": True,
        "channel_sha256": hashlib.sha256(handle.encode()).hexdigest(),
        "receiver_collective_completed": future is not None,
        "scope": "transport_and_two_selected_gen_sentinels",
    }
    adapter._bagel_corl_gen_receipt = receipt
    return copy.deepcopy(receipt)


def receive_receipted_weights(worker, *, update_id, handle, use_shm, receiver_cls):
    """Publish a receipt only after the actual receiver/load/ACK loop succeeds."""
    worker._bagel_corl_gen_receipt = None
    if not isinstance(update_id, str) or not update_id:
        raise ValueError("GEN receipt requires an actual transport update ID")
    pipeline = getattr(getattr(worker, "model_runner", None), "pipeline", None)
    native_sentinels(pipeline, include_digests=False)  # layout only before loading
    if getattr(worker, "lora_manager", None) is not None:
        raise ValueError("GEN receipts require merged weights without active serving adapters")
    receipt = {
        "update_id": update_id,
        "tensor_count": 0,
        "byte_count": 0,
        "bucket_count": 0,
        "channel_sha256": hashlib.sha256(handle.encode()).hexdigest(),
    }
    names, accepted = set(), set()
    terminal = False

    def load_bucket(weights, is_last=False):
        nonlocal terminal
        if terminal:
            raise ValueError("GEN receipt received data after terminal bucket")
        receipt["bucket_count"] += 1
        for name, tensor in weights:
            if name in names:
                raise ValueError(f"Duplicate received GEN tensor: {name}")
            names.add(name)
            receipt["tensor_count"] += 1
            receipt["byte_count"] += tensor.numel() * tensor.element_size()
        loaded = pipeline.load_weights(weights) if weights else set()
        if not isinstance(loaded, set):
            raise ValueError("GEN native loader must report accepted parameter names")
        accepted.update(loaded)
        terminal = is_last

    receiver_cls(zmq_handle=handle, device=worker.device, use_shm=use_shm).receive_weights(
        on_bucket_received=load_bucket
    )
    if not terminal or not set(GEN_SENTINELS).issubset(names) or not set(GEN_SENTINELS.values()).issubset(accepted):
        raise ValueError("GEN receiver missing terminal bucket or accepted sentinels")
    receipt.update(
        native=native_sentinels(pipeline),
        accepted_parameter_count=len(accepted),
        terminal_acknowledged=True,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        scope="transport_and_two_selected_gen_sentinels",
    )
    worker._bagel_corl_gen_receipt = receipt
    return copy.deepcopy(receipt)


def get_receiver_receipt(worker):
    """Return current native evidence only for the last successfully completed stream."""
    receipt = getattr(worker, "_bagel_corl_gen_receipt", None)
    if receipt is None:
        return None
    result = copy.deepcopy(receipt)
    result["native"] = native_sentinels(worker.model_runner.pipeline)
    return result


def verify_publication_receipts(senders, receivers, *, expected_tp_size):
    """Require complete physical TP coverage and exact transport/sentinel joins.

    Call once per actual GEN replica; multi-replica callers must enumerate each
    replica separately. Logical server steps remain separate server observations.
    """
    if type(expected_tp_size) is not int or expected_tp_size < 1:
        raise ValueError("Invalid expected GEN TP topology")
    if len(senders) != expected_tp_size or len(receivers) != expected_tp_size:
        raise ValueError("Incomplete GEN sender/receiver TP coverage")
    channels = {item["channel_sha256"]: item for item in senders}
    if len(channels) != expected_tp_size or not any(s.get("receiver_collective_completed") for s in senders):
        raise ValueError("GEN publication missing unique channels or collective completion")
    updates = {item["update_id"] for item in senders}
    if len(updates) != 1:
        raise ValueError("GEN senders disagree on transport update ID")
    ranks, seen_channels = set(), set()
    for received in receivers:
        channel = received["channel_sha256"]
        sent = channels.get(channel)
        if sent is None or channel in seen_channels:
            raise ValueError("GEN receiver has unknown or duplicated transport channel")
        seen_channels.add(channel)
        if not sent.get("terminal_acknowledged") or not received.get("terminal_acknowledged"):
            raise ValueError("GEN publication lacks terminal transport acknowledgment")
        for key in ("update_id", "tensor_count", "byte_count"):
            if sent[key] != received[key]:
                raise ValueError(f"GEN transport receipt mismatch: {key}")
        native = received["native"]
        if native["tp_size"] != expected_tp_size or native["tp_rank"] in ranks:
            raise ValueError("GEN receiver TP topology mismatch")
        ranks.add(native["tp_rank"])
        verify_sentinels(sent, native)
    if ranks != set(range(expected_tp_size)):
        raise ValueError("Incomplete GEN physical TP coverage")
    return {
        "update_id": next(iter(updates)),
        "verified_tp_ranks": sorted(ranks),
        "scope": "transport_and_two_selected_gen_sentinels",
        "full_model_verified": False,
    }


def flatten_receiver_receipts(results, *, expected_workers):
    """Reject missing/error/duplicate native worker responses from collective RPC."""
    records = []

    def visit(value):
        if isinstance(value, list | tuple):
            for item in value:
                visit(item)
        elif isinstance(value, dict) and value.get("terminal_acknowledged") is True and value.get("native"):
            records.append(value)
        else:
            raise ValueError("GEN receiver receipt missing or unsupported")

    visit(results)
    if len(records) != expected_workers:
        raise ValueError("GEN receipt did not return all physical TP workers")
    identities = [(item.get("hostname"), item.get("pid")) for item in records]
    if (
        any(not host or type(pid) is not int or pid <= 0 for host, pid in identities)
        or len(set(identities)) != expected_workers
    ):
        raise ValueError("GEN receipt missing or duplicated physical process identity")
    if (
        {r["native"]["tp_rank"] for r in records} != set(range(expected_workers))
        or any(r["native"]["tp_size"] != expected_workers for r in records)
        or len({r["update_id"] for r in records}) != 1
    ):
        raise ValueError("GEN receipts disagree on physical TP topology or update")
    return records
