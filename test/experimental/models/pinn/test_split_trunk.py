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

"""Tests for SplitTrunkNet (MOD-008a/b/c, MOD-009)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.pinn.split_trunk import SplitTrunkNet

_DATA_DIR = Path(__file__).parent / "data"
_REF = _DATA_DIR / "split_trunk_ref.pth"
_SEED = 0


def _build(config):
    if config == "default":
        # single group over all inputs/outputs
        return SplitTrunkNet(
            in_features=3,
            out_features=2,
            subnet_kwargs={"layer_size": 16, "num_layers": 2},
        )
    return SplitTrunkNet(
        in_features=3,
        out_features=2,
        key_mapping={"u": [0, 1], "v": [2]},
        output_groups={"u": [0], "v": [1]},
        positive_keys=[1],
        subnet_kwargs={"layer_size": 8, "num_layers": 2},
    )


# MOD-008a -----------------------------------------------------------------
@pytest.mark.parametrize("config", ["default", "custom"])
def test_constructor(config):
    torch.manual_seed(_SEED)
    model = _build(config)
    if config == "default":
        assert list(model.key_mapping) == ["all"]
        assert len(model.subnets) == 1
        assert model.positive_keys == []
    else:
        assert set(model.key_mapping) == {"u", "v"}
        assert len(model.subnets) == 2
        assert model.positive_keys == [1]

    x = torch.randn(16, 3)
    out = model(x)
    assert out.shape == (16, 2)
    assert sum(p.numel() for p in model.parameters()) > 0


# MOD-009: subnet is injected as a class, not a string
def test_mod009_factory_injection():
    from physicsnemo.models.mlp.fully_connected import FullyConnected

    model = SplitTrunkNet(
        in_features=2,
        out_features=1,
        subnet_cls=FullyConnected,
        subnet_kwargs={"layer_size": 8, "num_layers": 2},
    )
    assert model.subnet_cls is FullyConnected
    assert model(torch.randn(8, 2)).shape == (8, 1)


def test_positivity_enforced():
    torch.manual_seed(_SEED)
    model = _build("custom")
    out = model(torch.randn(64, 3))
    # output index 1 is softplus-positive
    assert (out[:, 1] > 0).all()


def test_invalid_mapping():
    with pytest.raises(ValueError, match="out of range"):
        SplitTrunkNet(
            in_features=2,
            out_features=1,
            key_mapping={"u": [5]},
            output_groups={"u": [0]},
        )


def test_invalid_input():
    torch.manual_seed(_SEED)
    model = _build("default")
    with pytest.raises(ValueError, match="Expected input"):
        model(torch.randn(8, 5))


# MOD-008b -----------------------------------------------------------------
def test_non_regression():
    torch.manual_seed(_SEED)
    model = _build("custom")
    torch.manual_seed(_SEED)
    x = torch.randn(16, 3)
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
    model = _build("custom")
    x = torch.randn(16, 3)
    out_before = model(x).detach()

    ckpt = tmp_path / "split_trunk.mdlus"
    model.save(str(ckpt))
    loaded = Module.from_checkpoint(str(ckpt))
    out_after = loaded(x).detach()

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)
    assert set(loaded.key_mapping) == set(model.key_mapping)
    assert loaded.positive_keys == model.positive_keys
