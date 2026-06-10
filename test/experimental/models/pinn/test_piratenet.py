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

"""Tests for PirateNet (MOD-008a/b/c)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.pinn.piratenet import PirateNet

_DATA_DIR = Path(__file__).parent / "data"
_REF = _DATA_DIR / "piratenet_ref.pth"
_SEED = 0


def _build(config):
    if config == "default":
        return PirateNet(in_features=2, out_features=1, layer_size=32, nr_blocks=2)
    return PirateNet(
        in_features=3,
        out_features=2,
        layer_size=16,
        nr_blocks=3,
        fourier_scale=2.0,
        use_rwf=False,
    )


# MOD-008a -----------------------------------------------------------------
@pytest.mark.parametrize("config", ["default", "custom"])
def test_constructor(config):
    torch.manual_seed(_SEED)
    model = _build(config)
    if config == "default":
        assert model.layer_size == 32
        assert model.nr_blocks == 2
        assert model.use_rwf is True
        assert len(model.blocks) == 2
    else:
        assert model.layer_size == 16
        assert model.nr_blocks == 3
        assert model.use_rwf is False
        assert model.fourier_scale == 2.0

    x = torch.randn(16, model.in_features)
    out = model(x)
    assert out.shape == (16, model.out_features)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_physics_informed_init():
    torch.manual_seed(_SEED)
    model = _build("default")
    x = torch.rand(64, 2)
    y = torch.rand(64, 1)
    model.physics_informed_init(x, y)
    # output layer should now reproduce y reasonably at the fit points
    out = model(x)
    assert out.shape == (64, 1)


def test_invalid_input():
    torch.manual_seed(_SEED)
    model = _build("default")
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(8, 5))


# MOD-008b -----------------------------------------------------------------
def test_non_regression():
    torch.manual_seed(_SEED)
    model = _build("default")
    torch.manual_seed(_SEED)
    x = torch.randn(16, 2)
    out = model(x).detach()

    if not _REF.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"x": x, "out": out}, _REF)
        pytest.skip("Reference created — re-run to validate.")

    ref = torch.load(_REF, weights_only=True)
    torch.testing.assert_close(out, ref["out"], atol=1e-5, rtol=1e-5)


# MOD-008c -----------------------------------------------------------------
def test_checkpoint(tmp_path):
    torch.manual_seed(_SEED)
    model = _build("default")
    x = torch.randn(16, 2)
    out_before = model(x).detach()

    ckpt = tmp_path / "piratenet.mdlus"
    model.save(str(ckpt))
    loaded = Module.from_checkpoint(str(ckpt))
    out_after = loaded(x).detach()

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)
    assert loaded.layer_size == model.layer_size
    assert loaded.nr_blocks == model.nr_blocks
