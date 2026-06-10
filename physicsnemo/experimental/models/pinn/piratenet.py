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

r"""PirateNet: Physics-Informed Residual AdapTivE Network.

A random Fourier feature embedding feeds dual gating tensors :math:`(U, V)` and
a stack of adaptive residual blocks with trainable mixing coefficients
:math:`\alpha`, followed by a linear output layer.  Hidden linear layers
optionally use random weight factorization (RWF) for better loss-landscape
conditioning.  The output layer can be warm-started with a least-squares fit via
:meth:`PirateNet.physics_informed_init`.

Based on `PirateNets: Physics-informed Deep Learning with Residual Adaptive
Networks <https://arxiv.org/abs/2402.00326>`_.  Random weight factorization
follows `Random weight factorization improves the training of continuous neural
representations <https://arxiv.org/abs/2210.01274>`_.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.nn import get_activation


class RWFLinear(nn.Module):
    r"""Linear layer with Random Weight Factorization (RWF).

    The weight matrix is factorised as :math:`W = \text{diag}(\exp(s)) \, V`,
    where :math:`V` is Glorot-initialised and :math:`s` is a per-row log-scale
    drawn from a log-normal distribution.  Storing the log-scale keeps the
    parametrisation numerically stable.

    Parameters
    ----------
    in_features : int
        Size of each input sample :math:`D_{in}`.
    out_features : int
        Size of each output sample :math:`D_{out}`.
    bias : bool, optional, default=True
        If ``True``, adds a learnable bias.
    mu : float, optional, default=1.0
        Mean of the log-normal distribution for the scale factors.
    sigma : float, optional, default=0.1
        Standard deviation of the log-normal distribution for the scale factors.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(\dots, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(\dots, D_{out})`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mu: float = 1.0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Glorot-initialised base weight
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.V)

        # per-row log-normal scale, stored as log for stability
        s_init = torch.empty(out_features)
        nn.init.normal_(s_init, mean=mu, std=sigma)
        self.log_s = nn.Parameter(torch.log(s_init.abs()))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the random-weight-factorized linear map.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(\dots, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(\dots, D_{out})`.
        """
        w = torch.exp(self.log_s).unsqueeze(-1) * self.V
        return nn.functional.linear(x, w, self.bias)


def _make_linear(
    in_features: int,
    out_features: int,
    use_rwf: bool,
    rwf_mu: float,
    rwf_sigma: float,
) -> nn.Module:
    r"""Build an RWF or standard linear layer.

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    use_rwf : bool
        Whether to use random weight factorization.
    rwf_mu : float
        RWF log-normal scale mean.
    rwf_sigma : float
        RWF log-normal scale standard deviation.

    Returns
    -------
    torch.nn.Module
        The constructed linear layer.
    """
    if use_rwf:
        return RWFLinear(
            in_features, out_features, bias=True, mu=rwf_mu, sigma=rwf_sigma
        )
    lin = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


class PirateNetBlock(nn.Module):
    r"""Single adaptive PirateNet residual block.

    Three dense sub-layers are gated against shared tensors :math:`(U, V)`, and
    the block output is mixed with its input through a trainable coefficient
    :math:`\alpha` (initialised to zero so the block starts as an identity map).

    Parameters
    ----------
    layer_size : int
        Width of every dense layer inside the block :math:`D`.
    activation : torch.nn.Module
        Point-wise activation module.
    use_rwf : bool, optional, default=True
        Replace :class:`torch.nn.Linear` with :class:`RWFLinear`.
    rwf_mu : float, optional, default=1.0
        RWF log-normal scale mean.
    rwf_sigma : float, optional, default=0.1
        RWF log-normal scale standard deviation.

    Forward
    -------
    x : torch.Tensor
        Block input of shape :math:`(B, D)`.
    u : torch.Tensor
        Gating tensor of shape :math:`(B, D)`.
    v : torch.Tensor
        Gating tensor of shape :math:`(B, D)`.

    Outputs
    -------
    torch.Tensor
        Block output of shape :math:`(B, D)`.
    """

    def __init__(
        self,
        layer_size: int,
        activation: nn.Module,
        use_rwf: bool = True,
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.fc1 = _make_linear(layer_size, layer_size, use_rwf, rwf_mu, rwf_sigma)
        self.fc2 = _make_linear(layer_size, layer_size, use_rwf, rwf_mu, rwf_sigma)
        self.fc3 = _make_linear(layer_size, layer_size, use_rwf, rwf_mu, rwf_sigma)

        # trainable residual coefficient, initialised so the block is identity
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        r"""Apply the gated residual block.

        Parameters
        ----------
        x : torch.Tensor
            Block input of shape :math:`(B, D)`.
        u : torch.Tensor
            Gating tensor of shape :math:`(B, D)`.
        v : torch.Tensor
            Gating tensor of shape :math:`(B, D)`.

        Returns
        -------
        torch.Tensor
            Block output of shape :math:`(B, D)`.
        """
        # first gated sub-layer
        f = self.activation(self.fc1(x))
        z1 = f * u + (1.0 - f) * v
        # second gated sub-layer
        g = self.activation(self.fc2(z1))
        z2 = g * u + (1.0 - g) * v
        # third dense sub-layer
        h = self.activation(self.fc3(z2))
        # adaptive residual connection
        return self.alpha * h + (1.0 - self.alpha) * x


class PirateNet(physicsnemo.Module):
    r"""Physics-Informed Residual AdapTivE Network (PirateNet).

    A random Fourier feature embedding feeds two gating tensors :math:`(U, V)`
    and a stack of :class:`PirateNetBlock` residual blocks, followed by a linear
    output layer.  Each residual block contains three dense layers, so the
    effective depth is :math:`3 \times \text{nr\_blocks}`.

    Parameters
    ----------
    in_features : int
        Number of input coordinate features :math:`D_{in}`.
    out_features : int
        Number of output features :math:`D_{out}`.
    layer_size : int, optional, default=256
        Hidden width shared by all layers.
    nr_blocks : int, optional, default=3
        Number of residual blocks.
    fourier_scale : float, optional, default=1.0
        Standard deviation of the random Gaussian matrix used for the Fourier
        feature embedding.  Larger values favour higher-frequency solutions.
    activation_fn : str, optional, default="tanh"
        Point-wise activation, resolved via
        :func:`~physicsnemo.nn.get_activation`.
    use_rwf : bool, optional, default=True
        Use random weight factorization on hidden linear layers.
    rwf_mu : float, optional, default=1.0
        RWF log-normal scale mean.
    rwf_sigma : float, optional, default=0.1
        RWF log-normal scale standard deviation.

    Forward
    -------
    x : torch.Tensor
        Coordinate tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.experimental.models.pinn.piratenet import PirateNet
    >>> model = PirateNet(in_features=2, out_features=1, layer_size=32, nr_blocks=2)
    >>> x = torch.randn(32, 2)
    >>> model(x).shape
    torch.Size([32, 1])
    """

    __model_checkpoint_version__ = "0.1.0"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_size: int = 256,
        nr_blocks: int = 3,
        fourier_scale: float = 1.0,
        activation_fn: str = "tanh",
        use_rwf: bool = True,
        rwf_mu: float = 1.0,
        rwf_sigma: float = 0.1,
    ) -> None:
        super().__init__(meta=ModelMetaData(func_torch=True, auto_grad=True))

        self.in_features = in_features
        self.out_features = out_features
        self.layer_size = layer_size
        self.nr_blocks = nr_blocks
        self.fourier_scale = fourier_scale
        self.activation_fn = activation_fn
        self.use_rwf = use_rwf

        self.activation = get_activation(activation_fn)

        # fixed random Fourier feature embedding -> 2m = layer_size features
        m = layer_size // 2
        b = torch.randn(m, in_features) * fourier_scale
        self.register_buffer("B", b)
        embed_dim = 2 * m

        # dual gate encoders and embedding projection
        self.enc_U = _make_linear(embed_dim, layer_size, use_rwf, rwf_mu, rwf_sigma)
        self.enc_V = _make_linear(embed_dim, layer_size, use_rwf, rwf_mu, rwf_sigma)
        self.embed_proj = _make_linear(
            embed_dim, layer_size, use_rwf, rwf_mu, rwf_sigma
        )

        # adaptive residual blocks
        self.blocks = nn.ModuleList(
            PirateNetBlock(
                layer_size=layer_size,
                activation=self.activation,
                use_rwf=use_rwf,
                rwf_mu=rwf_mu,
                rwf_sigma=rwf_sigma,
            )
            for _ in range(nr_blocks)
        )

        # linear output head
        self.output_layer = nn.Linear(layer_size, out_features, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def embed(self, x: Tensor) -> Tensor:
        r"""Return the random Fourier feature embedding of ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Coordinate tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape :math:`(B, 2m)` where :math:`2m` equals
            ``layer_size`` (rounded down).
        """
        proj = x @ self.B.T
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def physics_informed_init(
        self,
        x_data: Float[Tensor, "n in_features"],
        y_data: Float[Tensor, "n out_features"],
    ) -> None:
        r"""Warm-start the output layer with a least-squares fit.

        Solves a linear least-squares problem mapping the (activated, projected)
        Fourier embedding of ``x_data`` to ``y_data``, and copies the solution
        into the output layer weight and bias.

        Parameters
        ----------
        x_data : torch.Tensor
            Collocation points of shape :math:`(N, D_{in})`.
        y_data : torch.Tensor
            Target values of shape :math:`(N, D_{out})`.

        Returns
        -------
        None
        """
        self.eval()
        with torch.no_grad():
            phi = self.embed(x_data)
            phi_proj = self.activation(self.embed_proj(phi))

            # augment with a bias column to solve for weight and bias jointly
            ones = torch.ones(
                phi_proj.shape[0], 1, device=phi_proj.device, dtype=phi_proj.dtype
            )
            a = torch.cat([phi_proj, ones], dim=-1)

            sol = torch.linalg.lstsq(a, y_data).solution
            self.output_layer.weight.copy_(sol[:-1].T)
            self.output_layer.bias.copy_(sol[-1])
        self.train()

    def forward(
        self, x: Float[Tensor, "batch in_features"]
    ) -> Float[Tensor, "batch out_features"]:
        r"""Forward pass through PirateNet.

        Parameters
        ----------
        x : torch.Tensor
            Coordinate tensor of shape :math:`(B, D_{in})`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(B, D_{out})`.
        """
        if not torch.compiler.is_compiling():
            if x.ndim < 2 or x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected input of shape (B, {self.in_features}), "
                    f"got tensor of shape {tuple(x.shape)}"
                )

        # Fourier embedding and shared gate tensors
        phi = self.embed(x)  # (B, layer_size)
        u = self.activation(self.enc_U(phi))  # (B, layer_size)
        v = self.activation(self.enc_V(phi))  # (B, layer_size)

        # project embedding into the first block's input space
        h = self.activation(self.embed_proj(phi))  # (B, layer_size)

        # adaptive residual blocks
        for block in self.blocks:
            h = block(h, u, v)

        return self.output_layer(h)
