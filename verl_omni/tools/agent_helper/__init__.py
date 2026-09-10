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

"""Tool-agent helpers for Mode (2a) image-gen (not FunctionTool bodies).

``tools/image_gen.py`` + ``tools/trajectory/`` are the frozen sidecars and their
process-local state. This package is imported by ``ImageGenToolAgentLoop``:
force-first curriculum, premature-judge rewrite, teacher-forced Hermes, forced
Reflection. Lives under ``tools/`` so ``agent_loop.utils`` stays diffusion-only.
"""
