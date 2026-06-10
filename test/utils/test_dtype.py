# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest
import torch

from physicsnemo.utils.dtype import get_default_dtype, set_default_dtype


@pytest.fixture(autouse=True)
def restore_default_dtype():
    """Restore the original global default dtype after each test."""
    original = torch.get_default_dtype()
    yield
    torch.set_default_dtype(original)


def test_set_default_dtype_float64():
    set_default_dtype(torch.float64)
    assert torch.get_default_dtype() == torch.float64
    assert get_default_dtype() == torch.float64


def test_set_default_dtype_float32():
    set_default_dtype(torch.float32)
    assert torch.get_default_dtype() == torch.float32
    assert get_default_dtype() == torch.float32


def test_set_default_dtype_default_is_float64():
    set_default_dtype()
    assert torch.get_default_dtype() == torch.float64


def test_set_default_dtype_rejects_integer_dtype():
    with pytest.raises(ValueError):
        set_default_dtype(torch.int64)


def test_set_default_dtype_rejects_non_dtype():
    with pytest.raises(ValueError):
        set_default_dtype("float64")
