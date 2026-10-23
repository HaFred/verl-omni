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

"""Exercise production receipt/transport code with CPU tensors and fake IPC boundaries."""

import ast
import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
ROLLOUT = ROOT / "verl_omni/workers/rollout/vllm_rollout"


def helper():
    path = ROLLOUT / "bagel_corl_gen_receipts.py"
    assert path.exists(), "GEN receipt foundation missing"
    spec = importlib.util.spec_from_file_location("gen_receipts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights():
    return {
        "transformer.layers.0.self_attn.o_proj_moe_gen.weight": torch.arange(32).reshape(4, 8).float(),
        "transformer.layers.1.mlp_moe_gen.down_proj.weight": torch.arange(48).reshape(4, 12).float(),
    }


def prepare(h, adapter):
    import hashlib

    records = [h.native_sentinels(pipeline(h, rank), include_digests=False) for rank in range(2)]
    for rank, record in enumerate(records):
        record["base_channel_sha256"] = hashlib.sha256(
            (adapter.zmq_handle if rank == 0 else adapter.zmq_handle + "-other").encode()
        ).hexdigest()
    h.prepare_adapter_capabilities(adapter, records)


def pipeline(h, rank=0, size=2):
    class UnquantizedLinearMethod:
        pass

    modules = {}
    parameters = {}
    for actor_name, native_name in h.GEN_SENTINELS.items():
        tensor = weights()[actor_name]
        width = tensor.shape[1] // size
        parameter = torch.nn.Parameter(tensor[:, rank * width : (rank + 1) * width].to(torch.bfloat16))
        parameter.input_dim = 1
        parameters[native_name] = parameter
        modules[native_name.rsplit(".", 1)[0]] = SimpleNamespace(
            tp_rank=rank,
            tp_size=size,
            input_size=tensor.shape[1],
            input_size_per_partition=width,
            quant_method=UnquantizedLinearMethod(),
        )
    language = SimpleNamespace(named_parameters=lambda: parameters.items(), get_submodule=modules.__getitem__)
    result = SimpleNamespace(bagel=SimpleNamespace(language_model=language), language_model=language)
    result.load_weights = lambda items: set(h.GEN_SENTINELS[n] for n, _ in items if n in h.GEN_SENTINELS)
    return result


def test_exact_mapping_and_each_actual_tp_slice_matches():
    h = helper()
    assert h.GEN_SENTINELS == {
        "transformer.layers.0.self_attn.o_proj_moe_gen.weight": "model.layers.0.self_attn.o_proj.gen_exp.weight",
        "transformer.layers.1.mlp_moe_gen.down_proj.weight": "model.layers.1.mlp_moe_gen.down_proj.weight",
    }
    expected = h.summarize_export(weights().items(), tp_size=2, dtype="bfloat16")
    for rank in range(2):
        observed = h.native_sentinels(pipeline(h, rank))
        h.verify_sentinels(expected, observed)
        assert observed["tp_rank"] == rank
    wrong = h.native_sentinels(pipeline(h, 1))
    wrong["tp_rank"] = 0
    with pytest.raises(ValueError, match="sentinel"):
        h.verify_sentinels(expected, wrong)


def test_merged_lora_delta_changes_only_affected_tp_shard():
    h = helper()
    original = weights()
    before = h.summarize_export(original.items(), tp_size=2, dtype="bfloat16")
    first = next(iter(original))
    original[first][:, :4] += torch.ones(4, 1) @ torch.ones(1, 4)
    after = h.summarize_export(original.items(), tp_size=2, dtype="bfloat16")
    assert before["sentinels"][first][0] != after["sentinels"][first][0]
    assert before["sentinels"][first][1] == after["sentinels"][first][1]


@pytest.mark.parametrize("problem", ["quant", "axis", "shape", "tp", "missing"])
def test_native_rejects_unsupported_layout(problem):
    h = helper()
    p = pipeline(h)
    language = p.language_model
    name = next(iter(h.GEN_SENTINELS.values()))
    param = dict(language.named_parameters())[name]
    module = language.get_submodule(name.rsplit(".", 1)[0])
    if problem == "quant":
        module.quant_method = object()
    elif problem == "axis":
        param.input_dim = 0
    elif problem == "shape":
        module.input_size += 1
    elif problem == "tp":
        module.tp_rank = 2
    else:
        language.named_parameters = lambda: []
    with pytest.raises(ValueError):
        h.native_sentinels(p)


@pytest.mark.parametrize("problem", ["duplicate", "missing", "dtype"])
def test_export_rejects_invalid_stream(problem):
    h = helper()
    items = list(weights().items())
    if problem == "duplicate":
        items.append(items[0])
    elif problem == "missing":
        items.pop()
    with pytest.raises(ValueError):
        h.summarize_export(items, tp_size=2, dtype="int8" if problem == "dtype" else "bfloat16")


def test_transport_waits_terminal_ack_then_receiver_before_commit():
    h = helper()
    events = []
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            dtype="bfloat16",
            tensor_model_parallel_size=2,
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1),
        ),
        zmq_handle="ipc:///tmp/base.sock",
        use_shm=False,
        _has_server=False,
        _bagel_corl_gen_receipt={"stale": True},
    )

    class Sender:
        def __init__(self, **kwargs):
            assert "update-step-7-seq-0" in kwargs["zmq_handle"]

        async def async_send_weights(self, stream):
            assert adapter._bagel_corl_gen_receipt is None
            list(stream)
            events.append("terminal_ack")

    async def execute(method, non_block, kwargs):
        assert method == "update_weights_from_ipc"
        assert kwargs["zmq_update_id"] == "step-7-seq-0"
        assert kwargs["bagel_corl_gen_receipt"] is True
        events.append("receiver_started")

        async def complete():
            assert events[-1] == "terminal_ack"
            events.append("receiver_completed")

        return complete()

    adapter._execute_method = execute
    prepare(h, adapter)
    receipt = asyncio.run(
        h.send_receipted_weights(
            adapter,
            weights().items(),
            global_steps=7,
            sender_cls=Sender,
            make_id=lambda step, seq: f"step-{step}-seq-{seq}",
            make_handle=lambda base, update: base + "-update-" + update,
        )
    )
    assert events == ["receiver_started", "terminal_ack", "receiver_completed"]
    assert receipt["tensor_count"] == 2
    assert receipt["byte_count"] == 320
    assert receipt["terminal_acknowledged"] is True
    assert adapter._bagel_corl_gen_receipt == receipt


def test_receiver_counts_actual_loads_and_invalidates_after_failure():
    h = helper()
    p = pipeline(h)
    worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=p), device=torch.device("cpu"))
    items = list(weights().items())
    events = []

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            assert worker._bagel_corl_gen_receipt is None
            on_bucket_received(items[:1], False)
            on_bucket_received(items[1:], True)
            events.append("terminal_ack")

    receipt = h.receive_receipted_weights(
        worker, update_id="step-7-seq-0", handle="ipc://own", use_shm=False, receiver_cls=Receiver
    )
    assert events == ["terminal_ack"]
    assert receipt["tensor_count"] == 2
    assert receipt["bucket_count"] == 2
    assert receipt["byte_count"] == 320
    assert receipt["terminal_acknowledged"] is True
    assert receipt["native"]["tp_rank"] == 0
    p.load_weights = lambda items: set()
    with pytest.raises(ValueError, match="accepted"):
        h.receive_receipted_weights(
            worker, update_id="step-7-seq-1", handle="ipc://own", use_shm=False, receiver_cls=Receiver
        )
    assert worker._bagel_corl_gen_receipt is None


def test_adapter_opt_out_calls_actual_super_method_unchanged(monkeypatch):
    helper()
    tree = ast.parse((ROLLOUT / "vllm_omni_async_server.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniServerAdapter")
    method = next((n for n in node.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_weights"), None)
    assert method is not None, "missing opt-in adapter override"
    calls = []

    class Base:
        async def update_weights(self, weights, global_steps=None, wire_format="named_tensors", **kwargs):
            calls.append((weights, global_steps, wire_format, kwargs))
            return "stock"

    import os

    scope = {"ServerAdapter": Base, "os": os}
    extracted = ast.ClassDef(
        name=node.name,
        bases=[ast.Name(id="ServerAdapter", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), "adapter", "exec"), scope)
    monkeypatch.delenv("BAGEL_CORL_GEN_RECEIPTS", raising=False)
    stream = iter([])
    adapter = scope[node.name]()
    adapter._bagel_corl_gen_receipt = {"stale": True}
    assert asyncio.run(adapter.update_weights(stream, global_steps=3, base_sync_done=True)) == "stock"
    assert calls == [(stream, 3, "named_tensors", {"base_sync_done": True})]
    assert adapter._bagel_corl_gen_receipt is None


@pytest.mark.parametrize(
    "field,value",
    [("ulysses_degree", 2), ("ring_degree", 2), ("pipeline_model_parallel_size", 2), ("cfg_parallel_size", 2)],
)
def test_sender_rejects_unsupported_parallel_layout_before_start(field, value):
    h = helper()
    config = SimpleNamespace(dtype="bfloat16", tensor_model_parallel_size=2, **{field: value})
    adapter = SimpleNamespace(config=config, _bagel_corl_gen_receipt={"stale": True})
    with pytest.raises(ValueError, match="parallel"):
        asyncio.run(
            h.send_receipted_weights(
                adapter, weights().items(), global_steps=0, sender_cls=None, make_id=None, make_handle=None
            )
        )
    assert adapter._bagel_corl_gen_receipt is None


def test_publication_join_rejects_missing_worker_wrong_channel_and_counts():
    h = helper()
    expected = h.summarize_export(weights().items(), tp_size=2, dtype="bfloat16")
    senders, receivers = [], []
    for rank in range(2):
        common = {
            "update_id": "step-0-seq-0",
            "channel_sha256": f"channel{rank}",
            "tensor_count": 2,
            "byte_count": 320,
            "terminal_acknowledged": True,
        }
        senders.append({**expected, **common, "receiver_collective_completed": rank == 0})
        receivers.append({**common, "native": h.native_sentinels(pipeline(h, rank))})
    assert h.verify_publication_receipts(senders, receivers, expected_tp_size=2)["verified_tp_ranks"] == [0, 1]
    with pytest.raises(ValueError):
        h.verify_publication_receipts(senders, receivers[:1], expected_tp_size=2)
    receivers[1]["byte_count"] += 1
    with pytest.raises(ValueError):
        h.verify_publication_receipts(senders, receivers, expected_tp_size=2)
    receivers[1]["byte_count"] -= 1
    receivers[1]["channel_sha256"] = "wrong"
    with pytest.raises(ValueError):
        h.verify_publication_receipts(senders, receivers, expected_tp_size=2)


def test_receiver_failure_before_terminal_ack_invalidates_success():
    h = helper()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(pipeline=pipeline(h)),
        device=torch.device("cpu"),
        _bagel_corl_gen_receipt={"stale": True},
    )

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received(list(weights().items()), True)
            raise RuntimeError("terminal ACK failed")

    with pytest.raises(RuntimeError, match="ACK"):
        h.receive_receipted_weights(worker, update_id="new", handle="ipc://new", use_shm=False, receiver_cls=Receiver)
    assert worker._bagel_corl_gen_receipt is None


def test_native_receipt_rpc_is_present_and_reads_fresh_tensors():
    h = helper()
    tree = ast.parse((ROLLOUT / "utils.py").read_text(encoding="utf-8"))
    worker_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniColocateWorkerExtension"
    )
    assert any(isinstance(n, ast.FunctionDef) and n.name == "get_bagel_corl_gen_receipt" for n in worker_class.body)
    p = pipeline(h)
    worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=p), _bagel_corl_gen_receipt={"update_id": "real"})
    before = h.get_receiver_receipt(worker)
    param = next(iter(dict(p.language_model.named_parameters()).values()))
    with torch.no_grad():
        param.add_(1)
    after = h.get_receiver_receipt(worker)
    assert before["native"] != after["native"]
    worker._bagel_corl_gen_receipt = None
    assert h.get_receiver_receipt(worker) is None


def test_actual_receiver_entrypoint_uses_update_handle_and_forwards_completed_receipt(monkeypatch):
    h = helper()
    module_name = "verl_omni.workers.rollout.vllm_rollout.bagel_corl_gen_receipts"
    monkeypatch.setitem(sys.modules, module_name, h)
    seen = []

    class Receiver:
        def __init__(self, **kwargs):
            seen.append(kwargs["zmq_handle"])

        def receive_weights(self, on_bucket_received):
            on_bucket_received(list(weights().items()), True)

    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        SimpleNamespace(BucketedWeightReceiver=Receiver),
    )
    tree = ast.parse((ROLLOUT / "utils.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniColocateWorkerExtension")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "update_weights_from_ipc")
    scope = {"make_update_zmq_handle": lambda base, update: base + "-" + update}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "receiver", "exec"), scope)
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(pipeline=pipeline(h)),
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc://own",
    )
    receipt = scope[method.name](worker, zmq_update_id="actual-update", bagel_corl_gen_receipt=True)
    assert seen == ["ipc://own-actual-update"]
    assert receipt["update_id"] == "actual-update"
    assert receipt == worker._bagel_corl_gen_receipt
    # An ordinary later publication clears previously verified evidence before
    # even a device validation failure; no stale success can escape.
    worker.device = None
    monkeypatch.setenv("BAGEL_CORL_GEN_RECEIPTS", "1")
    with pytest.raises(RuntimeError, match="device"):
        scope[method.name](worker)
    assert worker._bagel_corl_gen_receipt is None


@pytest.mark.parametrize("failure", ["sender", "receiver", "cache"])
def test_sender_failure_invalidates_prior_receipt(failure):
    h = helper()
    adapter = SimpleNamespace(
        config=SimpleNamespace(
            dtype="bfloat16",
            tensor_model_parallel_size=2,
            checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1),
        ),
        zmq_handle="ipc://own",
        use_shm=False,
        _has_server=failure == "cache",
        _bagel_corl_gen_receipt={"stale": True},
    )

    class Sender:
        def __init__(self, **kwargs):
            pass

        async def async_send_weights(self, stream):
            list(stream)
            if failure == "sender":
                raise RuntimeError("sender")

    class Completion:
        def __await__(self):
            async def result():
                if failure == "receiver":
                    raise RuntimeError("receiver")

            return result().__await__()

    async def execute(*args, **kwargs):
        return Completion()

    async def clear():
        raise RuntimeError("cache")

    adapter._execute_method = execute
    adapter.server_handle = SimpleNamespace(clear_kv_cache=SimpleNamespace(remote=clear))
    prepare(h, adapter)
    with pytest.raises(RuntimeError, match=failure):
        asyncio.run(
            h.send_receipted_weights(
                adapter,
                weights().items(),
                global_steps=1,
                sender_cls=Sender,
                make_id=lambda *_: "new",
                make_handle=lambda *_: "ipc://new",
            )
        )
    assert adapter._bagel_corl_gen_receipt is None


@pytest.mark.parametrize("kind", ["sp", "cfg"])
def test_native_rejects_other_parallel_groups(kind):
    h = helper()
    p = pipeline(h)
    p.bagel._sp_size = 2 if kind == "sp" else 1
    p.bagel.parallel_config = SimpleNamespace(cfg_parallel_size=2 if kind == "cfg" else 1)
    with pytest.raises(ValueError, match="parallel"):
        h.native_sentinels(p)


def test_sender_getter_returns_independent_copy_and_blocking_all_rank_dispatch():
    path = ROOT / "verl_omni/workers/engine_workers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_bagel_gen_publication_receipts"
    ]
    assert len(matches) == 1, "missing actor-rank receipt getter"
    node = matches[0]
    decorator = ast.unparse(node.decorator_list[0])
    assert "Dispatch.ONE_TO_ALL" in decorator and "blocking=False" not in decorator
    node.decorator_list = []
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "getter", "exec"), scope)
    worker = SimpleNamespace(rollout=SimpleNamespace(_bagel_corl_gen_receipt={"sentinels": {"a": [1]}}))
    observed = scope[node.name](worker)
    observed["sentinels"]["a"].append(2)
    assert worker.rollout._bagel_corl_gen_receipt == {"sentinels": {"a": [1]}}


def test_server_collection_observes_step_and_targets_gen_stage_zero(monkeypatch):
    h = helper()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_gen_receipts", h)
    tree = ast.parse((ROLLOUT / "vllm_omni_async_server.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniHttpServer")
    node = next(
        (
            n
            for n in cls.body
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "get_bagel_corl_gen_publication_receipts"
        ),
        None,
    )
    assert node is not None, "missing serving receipt collection API"
    strategy = type("DiffusionStrategy", (), {})
    scope = {"asyncio": asyncio, "DiffusionStrategy": strategy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "server_getter", "exec"), scope)
    records = []
    for rank in range(2):
        records.append(
            {
                "update_id": "actual",
                "native": h.native_sentinels(pipeline(h, rank)),
                "hostname": "host",
                "pid": 100 + rank,
                "terminal_acknowledged": True,
            }
        )
    calls = []

    async def rpc(**kwargs):
        calls.append(kwargs)
        return [records]

    server = SimpleNamespace(
        node_rank=0,
        _generate_strategy=strategy(),
        replica_world_size=2,
        replica_rank=9,
        global_steps=17,
        engine=SimpleNamespace(request_states={}, default_sampling_params_list=[None], collective_rpc=rpc),
    )
    result = asyncio.run(scope[node.name](server))
    assert calls == [{"method": "get_bagel_corl_gen_receipt", "stage_ids": [0]}]
    assert result["observed_global_steps"] == 17
    assert result["server_replica_id"] == 9
    assert result["worker_receipts"] == records
    assert result["publication_verified"] is False
    assert server._bagel_corl_fixed_scoring is False
    server.engine.request_states = {"running": object()}
    with pytest.raises(RuntimeError, match="quiescent"):
        asyncio.run(scope[node.name](server))


def test_native_preflight_available_before_any_update(monkeypatch):
    h = helper()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_gen_receipts", h)
    tree = ast.parse((ROLLOUT / "utils.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniColocateWorkerExtension")
    method = next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_bagel_corl_gen_sentinels"), None
    )
    assert method is not None, "native capability preflight missing"
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "preflight", "exec"), scope)
    worker = SimpleNamespace(model_runner=SimpleNamespace(pipeline=pipeline(h)), _get_zmq_handle=lambda: "ipc://own")
    monkeypatch.setattr(h, "_digest", lambda *_: (_ for _ in ()).throw(AssertionError("must not read sleeping values")))
    assert scope[method.name](worker)["tp_size"] == 2
    assert not hasattr(worker, "_bagel_corl_gen_receipt")
    worker.lora_manager = object()
    with pytest.raises(ValueError, match="adapter"):
        scope[method.name](worker)


def test_preflight_requires_all_native_tp_workers_before_any_send():
    h = helper()
    records = [h.native_sentinels(pipeline(h, rank), include_digests=False) for rank in range(2)]
    assert h.validate_native_capabilities([records], expected_tp_size=2, dtype="bfloat16") == records
    with pytest.raises(ValueError):
        h.validate_native_capabilities(records[:1], expected_tp_size=2, dtype="bfloat16")
    with pytest.raises(ValueError):
        h.validate_native_capabilities([records[0], records[0]], expected_tp_size=2, dtype="bfloat16")
    with pytest.raises(ValueError):
        h.validate_native_capabilities(records, expected_tp_size=2, dtype="float32")
    with pytest.raises(ValueError):
        h.validate_native_capabilities([], expected_tp_size=0, dtype="bfloat16")


def test_actual_adapter_opt_in_forwards_marker_and_records_completion(monkeypatch):
    h = helper()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_gen_receipts", h)
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.workers.rollout.vllm_rollout.zmq_utils",
        SimpleNamespace(
            make_update_zmq_id=lambda step, seq: f"step-{step}-seq-{seq}",
            make_update_zmq_handle=lambda base, update: base + "-" + update,
        ),
    )
    events = []

    class Sender:
        def __init__(self, **kwargs):
            assert kwargs["zmq_handle"] == "ipc://own-step-0-seq-0"

        async def async_send_weights(self, stream):
            list(stream)
            events.append("ack")

    monkeypatch.setitem(
        sys.modules,
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        SimpleNamespace(BucketedWeightSender=Sender),
    )
    tree = ast.parse((ROLLOUT / "vllm_omni_async_server.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "vLLMOmniServerAdapter")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_weights")
    import os

    scope = {"ServerAdapter": object, "os": os}
    extracted = ast.ClassDef(
        name=cls.name,
        bases=[ast.Name(id="ServerAdapter", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), "adapter", "exec"), scope)
    adapter = scope[cls.name]()
    adapter.config = SimpleNamespace(
        dtype="bfloat16",
        tensor_model_parallel_size=2,
        checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=1),
    )
    adapter.zmq_handle, adapter.use_shm, adapter._has_server = "ipc://own", False, True

    async def execute(method, non_block, kwargs):
        assert kwargs["bagel_corl_gen_receipt"] is True
        assert kwargs["base_sync_done"] is True

        async def complete():
            events.append("receiver")

        return complete()

    async def clear():
        events.append("cache")

    async def stamp(step):
        assert step == 0
        events.append("stamp")

    adapter._execute_method = execute
    adapter.server_handle = SimpleNamespace(
        clear_kv_cache=SimpleNamespace(remote=clear), set_global_steps=SimpleNamespace(remote=stamp)
    )
    prepare(h, adapter)
    monkeypatch.setenv("BAGEL_CORL_GEN_RECEIPTS", "1")
    result = asyncio.run(adapter.update_weights(weights().items(), global_steps=0, base_sync_done=True))
    assert result["update_id"] == "step-0-seq-0"
    assert events == ["ack", "receiver", "cache", "stamp"]


def test_adapter_requires_complete_matching_preflight_before_any_receiver_or_sender():
    h = helper()
    adapter = SimpleNamespace(
        config=SimpleNamespace(dtype="bfloat16", tensor_model_parallel_size=2),
        zmq_handle="ipc://own",
        _bagel_corl_gen_receipt={"old": True},
    )
    events = []

    async def execute(*args, **kwargs):
        events.append("receiver_started")
        raise AssertionError("must not start receiver")

    adapter._execute_method = execute
    with pytest.raises(ValueError, match="preflight"):
        asyncio.run(
            h.send_receipted_weights(
                adapter, weights().items(), global_steps=0, sender_cls=None, make_id=None, make_handle=None
            )
        )
    assert events == []
    assert adapter._bagel_corl_gen_receipt is None
    records = [h.native_sentinels(pipeline(h, rank), include_digests=False) for rank in range(2)]
    for record in records:
        record["base_channel_sha256"] = "wrong"
    with pytest.raises(ValueError, match="channel"):
        h.prepare_adapter_capabilities(adapter, records)
    assert adapter._bagel_corl_gen_receipt_capabilities is None


def test_prepare_rpc_requires_blocking_all_rank_gate(monkeypatch):
    tree = ast.parse((ROOT / "verl_omni/workers/engine_workers.py").read_text(encoding="utf-8"))
    node = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "prepare_bagel_gen_publication_receipts"
        ),
        None,
    )
    assert node is not None, "missing all-rank preflight gate RPC"
    decorator = ast.unparse(node.decorator_list[0])
    assert "Dispatch.ONE_TO_ALL" in decorator and "blocking=False" not in decorator
    h = helper()
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.bagel_corl_gen_receipts", h)
    node.decorator_list = []
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "prepare", "exec"), scope)
    adapter = SimpleNamespace(
        config=SimpleNamespace(dtype="bfloat16", tensor_model_parallel_size=2), zmq_handle="ipc://own", replica_rank=0
    )
    prepare(h, adapter)
    records = adapter._bagel_corl_gen_receipt_capabilities
    worker = SimpleNamespace(rollout=adapter)
    assert scope[node.name](worker, {0: records})["tp_size"] == 2
    with pytest.raises(ValueError):
        scope[node.name](worker, {0: records, 1: records[:1]})
    assert adapter._bagel_corl_gen_receipt_capabilities is None
