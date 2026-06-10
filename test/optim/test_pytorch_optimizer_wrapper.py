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

import importlib.util

import pytest
import torch

from physicsnemo.optim import pytorch_optimizer_wrapper as pow_wrap
from physicsnemo.optim.pytorch_optimizer_wrapper import (
    SOAP,
    make_pytorch_optimizer,
)

_HAS_PYTORCH_OPTIMIZER = importlib.util.find_spec("pytorch_optimizer") is not None


def test_module_imports_without_dependency():
    # Importing the module must never fail, regardless of the optional dep.
    assert hasattr(pow_wrap, "SOAP")
    assert hasattr(pow_wrap, "make_pytorch_optimizer")


@pytest.mark.skipif(
    not _HAS_PYTORCH_OPTIMIZER, reason="pytorch_optimizer not installed"
)
def test_soap_runs_one_step():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = SOAP(model.parameters(), lr=3e-3)

    x = torch.randn(8, 4)
    opt.zero_grad()
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


@pytest.mark.skipif(
    not _HAS_PYTORCH_OPTIMIZER, reason="pytorch_optimizer not installed"
)
def test_make_pytorch_optimizer_builds_soap():
    model = torch.nn.Linear(4, 4)
    opt = make_pytorch_optimizer("SOAP", model.parameters(), lr=3e-3)
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.skipif(_HAS_PYTORCH_OPTIMIZER, reason="pytorch_optimizer is installed")
def test_helpful_error_when_dependency_absent():
    model = torch.nn.Linear(4, 4)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        SOAP(model.parameters())
    msg = str(exc_info.value)
    assert "pytorch_optimizer" in msg
    assert "pip install" in msg

    with pytest.raises(ModuleNotFoundError):
        make_pytorch_optimizer("SOAP", model.parameters())


def test_error_message_format_is_actionable():
    # The install hint must mention both install paths regardless of dep state.
    hint = pow_wrap._INSTALL_HINT
    assert "nvidia-physicsnemo[optim]" in hint
    assert "pip install pytorch_optimizer" in hint
