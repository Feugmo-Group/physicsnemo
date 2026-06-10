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

"""Tests for FiniteBasisNet (MOD-008a/b/c)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.pinn.fb_pinn import FiniteBasisNet

_DATA_DIR = Path(__file__).parent / "data"
_REF = _DATA_DIR / "fb_pinn_ref.pth"
_SEED = 0


def _build(config):
    if config == "default":
        return FiniteBasisNet(
            in_features=2, out_features=1, nr_domains=3, layer_size=16, nr_layers=2
        )
    return FiniteBasisNet(
        in_features=2,
        out_features=2,
        nr_domains=[2, 3],
        overlap_ratio=2.0,
        window_fn="sigmoid",
        layer_size=8,
        nr_layers=2,
        domain_bounds=[(0.0, 2.0), (-1.0, 1.0)],
    )


# MOD-008a -----------------------------------------------------------------
@pytest.mark.parametrize("config", ["default", "custom"])
def test_constructor(config):
    model = _build(config)
    if config == "default":
        assert model.window_fn == "cosine"
        assert model.overlap_ratio == 2.7
        assert model.total_subdomains == 9
        assert model.domain_bounds == ((0.0, 1.0), (0.0, 1.0))
    else:
        assert model.window_fn == "sigmoid"
        assert model.overlap_ratio == 2.0
        assert model.total_subdomains == 6
        assert model.out_features == 2

    x = torch.rand(16, 2)
    out = model(x)
    assert out.shape == (16, model.out_features)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_invalid_window():
    with pytest.raises(ValueError, match="window_fn"):
        FiniteBasisNet(in_features=2, out_features=1, window_fn="bogus")


def test_invalid_input():
    model = _build("default")
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.rand(8, 3))


# MOD-008b -----------------------------------------------------------------
def test_non_regression():
    torch.manual_seed(_SEED)
    model = _build("default")
    torch.manual_seed(_SEED)
    x = torch.rand(16, 2)
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
    x = torch.rand(16, 2)
    out_before = model(x).detach()

    ckpt = tmp_path / "fb_pinn.mdlus"
    model.save(str(ckpt))
    loaded = Module.from_checkpoint(str(ckpt))
    out_after = loaded(x).detach()

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)
    assert loaded.total_subdomains == model.total_subdomains
    assert loaded.window_fn == model.window_fn
