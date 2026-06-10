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

"""Optimizer utilities for PhysicsNeMo."""

from physicsnemo.optim.aggregators import (
    AGGREGATOR_NAMES,
    Aggregator,
    BalancedResidualDecayRate,
    build_aggregator,
)
from physicsnemo.optim.combined_optimizer import CombinedOptimizer
from physicsnemo.optim.lbfgs_phase import TwoPhaseOptimizer
from physicsnemo.optim.loss_landscape import loss_landscape_scan, plot_landscape
from physicsnemo.optim.muon_adam import MuonAdam
from physicsnemo.optim.natural_gradient import (
    NaturalGradient,
    SketchedNaturalGradient,
    compute_gram,
    compute_gram_functional,
)
from physicsnemo.optim.pytorch_optimizer_wrapper import SOAP, make_pytorch_optimizer

__all__ = [
    "CombinedOptimizer",
    "TwoPhaseOptimizer",
    "loss_landscape_scan",
    "plot_landscape",
    "AGGREGATOR_NAMES",
    "Aggregator",
    "BalancedResidualDecayRate",
    "build_aggregator",
    "MuonAdam",
    "NaturalGradient",
    "SketchedNaturalGradient",
    "compute_gram",
    "compute_gram_functional",
    "SOAP",
    "make_pytorch_optimizer",
]
