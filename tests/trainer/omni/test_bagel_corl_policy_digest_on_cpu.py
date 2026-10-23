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

"""Frozen-policy hashes cover actual local parameters without gathering or mutation."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def runtime():
    spec = importlib.util.spec_from_file_location("policy_digest", ROOT / "verl_omni/workers/bagel_corl_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_digest_covers_frozen_and_trainable_values_and_preserves_execution_state():
    module = torch.nn.Module()
    module.register_parameter(
        "base", torch.nn.Parameter(torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), requires_grad=False)
    )
    module.register_parameter("adapter", torch.nn.Parameter(torch.tensor(0.25)))
    module.adapter.grad = torch.tensor(3.0)
    rng = torch.get_rng_state().clone()
    pointers = [p.data_ptr() for p in module.parameters()]
    result = runtime().policy_parameter_digest(module)
    assert result["parameter_count"] == 2
    assert result["local_numel"] == 7
    assert result["trainable_local_numel"] == 1
    assert result["scope"] == "rank_local_parameters_only"
    assert len(result["parameters_sha256"]) == 64
    assert result == runtime().policy_parameter_digest(module)
    assert torch.equal(rng, torch.get_rng_state())
    assert pointers == [p.data_ptr() for p in module.parameters()]
    assert module.adapter.grad.item() == 3.0
    with torch.no_grad():
        module.base[0, 0] += 1
    assert result["parameters_sha256"] != runtime().policy_parameter_digest(module)["parameters_sha256"]


def test_digest_covers_names_dtype_shape_and_is_independent_of_registration_order():
    def model(reverse=False, dtype=torch.float32, shape=(2,)):
        result = torch.nn.Module()
        pairs = [("a", torch.ones(shape, dtype=dtype)), ("b", torch.zeros(1))]
        for name, value in reversed(pairs) if reverse else pairs:
            result.register_parameter(name, torch.nn.Parameter(value))
        return result

    digest = runtime().policy_parameter_digest
    baseline = digest(model())["parameters_sha256"]
    assert baseline == digest(model(reverse=True))["parameters_sha256"]
    assert baseline != digest(model(dtype=torch.float64))["parameters_sha256"]
    assert baseline != digest(model(shape=(1, 2)))["parameters_sha256"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_policy_parameters_reject(value):
    module = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(value)
    with pytest.raises(ValueError, match="Nonfinite"):
        runtime().policy_parameter_digest(module)


def test_engine_rpc_hashes_the_live_actor_engine_module_and_guards_other_roles():
    tree = ast.parse((ROOT / "verl_omni/workers/engine_workers.py").read_text(encoding="utf-8"))
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "get_bagel_policy_digest")
    assert any("ONE_TO_ALL" in ast.unparse(d) for d in method.decorator_list)
    method.decorator_list = []
    namespace = {"policy_parameter_digest": runtime().policy_parameter_digest}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "engine_rpc", "exec"), namespace
    )
    model = torch.nn.Linear(2, 1)
    worker = SimpleNamespace(
        role="actor",
        config=SimpleNamespace(model={"composite_mode": "bagel_corl"}),
        actor=SimpleNamespace(engine=SimpleNamespace(module=model)),
    )
    assert namespace[method.name](worker) == runtime().policy_parameter_digest(model)
    worker.config.model["composite_mode"] = "other"
    with pytest.raises(ValueError, match="bagel_corl"):
        namespace[method.name](worker)
