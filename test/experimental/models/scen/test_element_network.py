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

"""Tests for SCENElementNetwork.

Covers MOD-008a (constructor/attributes), MOD-008b (non-regression),
MOD-008c (checkpoint round-trip).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.models.scen.element_network import SCENElementNetwork

_DATA_DIR = Path(__file__).parent / "data"
_GOLDEN_SCEN = _DATA_DIR / "scen_element_network_v1.pth"
_SEED = 42

_TWO_ELEMENT_CFG = [
    {"N": 8, "a": 0.0, "b": 0.5},
    {"N": 8, "a": 0.5, "b": 1.0},
]

_MIXED_N_CFG = [
    {"N": 6, "a": 0.0, "b": 0.3, "alpha": 0.5},
    {"N": 10, "a": 0.3, "b": 1.0},
]


# ---------------------------------------------------------------------------
# MOD-008a: constructor and attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg, hidden_dim, n_layers, backbone",
    [
        (_TWO_ELEMENT_CFG, 16, 2, "mlp"),
        (_TWO_ELEMENT_CFG, 8, 3, "mlp"),
        (_TWO_ELEMENT_CFG, 8, 2, "kan"),
    ],
)
def test_scen_constructor(cfg, hidden_dim, n_layers, backbone):
    """SCENElementNetwork constructs with expected attributes."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(
        cfg, hidden_dim=hidden_dim, n_layers=n_layers, backbone=backbone
    )
    assert model.element_sizes == [c["N"] for c in cfg]
    assert model.hidden_dim == hidden_dim
    assert model.n_layers == n_layers
    assert model.backbone == backbone
    assert len(model.mappers) == len(cfg)


def test_scen_invalid_backbone():
    with pytest.raises(ValueError, match="backbone must be 'mlp' or 'kan'"):
        SCENElementNetwork(_TWO_ELEMENT_CFG, backbone="rnn")


def test_scen_buffers_registered():
    """D1_global, D2_global, D4_global, w_global are registered buffers."""
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    buffer_names = [n for n, _ in model.named_buffers()]
    for expected in ("D1_global", "D2_global", "D4_global", "w_global"):
        assert expected in buffer_names, f"Buffer '{expected}' not registered"


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg", [_TWO_ELEMENT_CFG, _MIXED_N_CFG])
def test_scen_forward_shape(cfg):
    """forward() returns tensor of shape (sum of N_k,)."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(cfg, hidden_dim=8, n_layers=2)
    u = model()
    expected_len = sum(c["N"] for c in cfg)
    assert u.shape == (expected_len,), f"Got {u.shape}, expected ({expected_len},)"


def test_scen_split_global():
    """split_global splits solution into per-element tensors."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    u = model()
    parts = model.split_global(u)
    assert len(parts) == 2
    assert parts[0].shape == (8,)
    assert parts[1].shape == (8,)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_scen_gradients_flow():
    """Gradients must propagate to sp (stacked ParameterDict) parameters."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    u = model()
    loss = u.pow(2).mean()
    loss.backward()
    for name, p in model.sp.named_parameters():
        assert p.grad is not None, f"No gradient for sp[{name}]"


def test_scen_networks_frozen():
    """self.networks parameters must have requires_grad=False."""
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    for name, p in model.networks.named_parameters():
        assert not p.requires_grad, f"networks param {name} is not frozen"


# ---------------------------------------------------------------------------
# sync_to_networks
# ---------------------------------------------------------------------------


def test_sync_to_networks():
    """After sync, networks[0] produces same output as functional_call path."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    # Train for a step to diverge weights
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        model().sum().backward()
        opt.step()

    model.sync_to_networks()
    # Now forward of networks[0] should match stacked-params path for element 0
    with torch.no_grad():
        u_stacked = model.split_global(model())[0]
        x_in = model._network_input(model.mappers[0])
        u_net = model.networks[0](x_in).squeeze(1)
    torch.testing.assert_close(u_stacked, u_net, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MOD-008b: non-regression
# ---------------------------------------------------------------------------


def _make_model_for_regression():
    torch.manual_seed(_SEED)
    return SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)


def test_scen_non_regression():
    """Forward output matches committed golden fixture."""
    model = _make_model_for_regression()
    with torch.no_grad():
        out = model()

    if not _GOLDEN_SCEN.exists():
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"output": out}, _GOLDEN_SCEN)
        pytest.skip("Golden fixture created — re-run to validate.")

    golden = torch.load(_GOLDEN_SCEN, weights_only=True)
    torch.testing.assert_close(out, golden["output"], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MOD-008c: checkpoint round-trip
# ---------------------------------------------------------------------------


def test_scen_checkpoint(tmp_path):
    """save() → from_checkpoint() reproduces identical forward output."""
    torch.manual_seed(_SEED)
    model = SCENElementNetwork(_TWO_ELEMENT_CFG, hidden_dim=8, n_layers=2)
    with torch.no_grad():
        out_before = model()

    ckpt_path = tmp_path / "scen.mdlus"
    model.save(str(ckpt_path))

    from physicsnemo import Module
    model_loaded = Module.from_checkpoint(str(ckpt_path))

    with torch.no_grad():
        out_after = model_loaded()

    torch.testing.assert_close(out_before, out_after, atol=1e-6, rtol=1e-6)
    assert model_loaded.element_sizes == model.element_sizes
    assert model_loaded.hidden_dim == model.hidden_dim
    assert model_loaded.backbone == model.backbone
