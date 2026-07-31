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
"""Allow padding-free training without flash_attn installed.

verl's ``attention_utils`` always imports ``flash_attn.bert_padding`` on CUDA.
Smoke / shared nodes often lack a flash-attn wheel; Transformers ships equivalent
helpers we can fall back to.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_attention_utils_flash_attn_fallback() -> None:
    try:
        import flash_attn  # noqa: F401

        return
    except ImportError:
        pass

    try:
        import verl.utils.attention_utils as attention_utils
        from einops import rearrange
        from transformers.modeling_flash_attention_utils import _index_first_axis, _pad_input, _unpad_input
    except ImportError as exc:
        logger.warning(
            "flash_attn not installed and Transformers padding helpers unavailable (%s); "
            "leaving verl.utils.attention_utils unchanged.",
            exc,
        )
        return

    def _get_attention_functions():
        from verl.utils.device import is_torch_npu_available

        if is_torch_npu_available(check_device=False):
            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, unpad_input
            from verl.utils.npu_flash_attn_utils import rearrange as npu_rearrange

            attention_utils._index_first_axis = index_first_axis
            attention_utils._pad_input = pad_input
            attention_utils._rearrange = npu_rearrange
            attention_utils._unpad_input = unpad_input
            return index_first_axis, pad_input, npu_rearrange, unpad_input

        attention_utils._index_first_axis = _index_first_axis
        attention_utils._pad_input = _pad_input
        attention_utils._rearrange = rearrange
        attention_utils._unpad_input = _unpad_input
        return _index_first_axis, _pad_input, rearrange, _unpad_input

    attention_utils._get_attention_functions = _get_attention_functions
    logger.warning(
        "flash_attn not installed; using transformers/einops padding helpers for "
        "verl.utils.attention_utils (smoke / sdpa path)."
    )
