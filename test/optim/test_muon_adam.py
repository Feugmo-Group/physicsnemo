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

import torch

from physicsnemo.optim.muon_adam import MuonAdam


def _make_model():
    torch.manual_seed(0)
    # Linear contributes a 2-D weight (ndim==2) and a 1-D bias (ndim==1).
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    )


def test_muon_adam_minimizes_quadratic():
    torch.manual_seed(0)
    model = _make_model()
    opt = MuonAdam(model.parameters(), lr=0.02)

    x = torch.randn(64, 8)
    target = (x.sum(dim=1, keepdim=True) * 0.5).detach()

    def loss_fn():
        return (model(x) - target).pow(2).mean()

    init_loss = loss_fn().item()
    final_loss = init_loss
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    # Loss must decrease substantially and reach a small value.
    assert final_loss < init_loss * 0.1, (init_loss, final_loss)
    assert final_loss < 1e-2, final_loss


def test_muon_adam_routes_params_by_ndim():
    model = _make_model()
    opt = MuonAdam(model.parameters(), lr=0.01)

    # Exactly two groups: muon (is_muon=True) and adam (is_muon=False).
    groups = {g["is_muon"]: g for g in opt.param_groups}
    assert set(groups) == {True, False}

    muon_group = groups[True]
    adam_group = groups[False]

    # Every param in the Muon group is 2-D or 4-D.
    assert all(p.ndim in (2, 4) for p in muon_group["params"])
    assert len(muon_group["params"]) == 2  # two Linear weights

    # Every param in the Adam group is not 2-D/4-D (here, the 1-D biases).
    assert all(p.ndim not in (2, 4) for p in adam_group["params"])
    assert len(adam_group["params"]) == 2  # two Linear biases
    assert all(p.ndim == 1 for p in adam_group["params"])


def test_muon_adam_step_returns_closure_loss():
    model = _make_model()
    opt = MuonAdam(model.parameters(), lr=0.01)
    x = torch.randn(16, 8)

    def closure():
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None
    assert torch.isfinite(loss)
