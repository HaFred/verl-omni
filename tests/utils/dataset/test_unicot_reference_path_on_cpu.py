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
"""CPU tests for UniCoT extra_info.reference_image_path stamping."""

from __future__ import annotations

import pandas as pd

from examples.agenticllmgrpo_trainer.bagel.stamp_unicot_reference_paths import stamp_reference_image_path


def test_stamp_reference_image_path():
    frame = pd.DataFrame(
        {
            "images": [["/data/ref.png", "/data/other.png"]],
            "extra_info": [{}],
        }
    )
    out = stamp_reference_image_path(frame)
    assert out.iloc[0]["extra_info"]["reference_image_path"] == "/data/ref.png"
