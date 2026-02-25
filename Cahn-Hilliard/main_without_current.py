from curses.textpad import rectangle
from pickletools import optimize
import numpy as np
from torch import Tensor
from torch.nn import Module
from torch import zeros_like as pt_zeros_like
from physicsnemo.sym.node import Node
import matplotlib.pyplot as plt
import sympy
import torch
from dataclasses import asdict
from sympy import Symbol, Number, Function, Eq, Integral, integrate
from physicsnemo.sym.eq.pde import PDE
import physicsnemo
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from timm.models.hrnet import cfg_cls
from physicsnemo.sym.models.activation import Activation
from torch.utils.checkpoint import detach_variable

import config
from config import register_custom_arch_configs
from models.custom_fullyconnected_conc import custom_FullyConnectedArch_conc
from models.custom_fullyconnected_eta import custom_FullyConnectedArch_eta
from models.custom_fullyconnected_phi import custom_FullyConnectedArch_phi
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry import Geometry
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.hydra.utils import compose, register_amp_configs, to_absolute_path
import torch
from scipy.constants import elementary_charge, Avogadro, k as Boltzmann
from geometry.custom_rec import custom_Rectangle
from geometry.custom_geometry import c_Geometry
from custom_inferencer import CustomInferencerPlotter
from eta_custom_inferencer import etaCustomInferencerPlotter
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)


i_a = 0         #-8e6: Applied current density [A/m²]
e = elementary_charge
NA = Avogadro
T = 1273.0     # Temperature 1273K

# initial set
charges = {
    "vac" : 2.0,
    "elec" : -1.0,
    "yzr" : -1.0,
}
vac = {
    "mu_YSZ" : 0.12 * e, # eV to J
    "mu_cathode" : 0.2 * e,
    "diffusivity" : 1.0e-8,
    "bulk_cathode_conc" : 83.0,
    "bulk_YSZ_conc" : 830.0,
    "rate_constant" : abs(i_a / e * NA * charges["vac"]),
}

elec = {
    "mu_YSZ" : 0.0 * e,
    "mu_cathode" : 0.0 * e,
    "diffusivity" : 2.0e-4,
    "bulk_cathode_conc": charges["vac"] * vac["bulk_cathode_conc"],
    "bulk_YSZ_conc" : 0.0,
    "rate_constant" : abs(i_a / e * NA * charges["elec"]),
}

yzr = {
    "mu_YSZ" : 0.0 * e,
    "mu_cathode" : 0.1 * e,
    "diffusivity" : 5.0e-20,
    "bulk_cathode_conc": 0.0,
    "bulk_YSZ_conc" : 1660.0,
    "rate_constant" : 0.0,
}

# Constants Using:
i_a = 0         #-8e6: Applied current density [A/m²]
kB = Boltzmann
e = elementary_charge
NA = Avogadro
d_ref = 1.0e-8 # m^2s^-1 vac["diffusivity"]
c_ref = 1660 # mol/m^3, reference concentration, yzr bulk YSZ concentration
L_ref = 2.5e-8 # m
t_ref = 6e-8 # keep 6e-8 for now, will be changed to L_ref^2 / D_ref 
rho_ref = 1.6e6 # from paper, C/m^3
phi_ref = (kB * T) / e # V approx 0.11, 0.1906986527
T = 1273.0     # Temperature 1273K
xmin = -2.5e-8 # m, domain size, 25 nm
xmax = 2.5e-8 # m, [-25, 25]
t = 6e-8        # time, s, 60 ns
epsilon_r = 40          # permittivity of free space
epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]

# Functions
def Lambda_c(eta):
    lc = (eta ** 2) * ((1 - eta) ** 2)
    return lc

def h(eta):
    r = (eta ** 3) * (6 * (eta ** 2) - 15 * eta + 10)
    return r

def mu_o_i(mu_cathode, mu_YSZ, eta):
    mu = mu_cathode + (mu_YSZ - mu_cathode) * h(eta)
    return mu

def rho(c_vac_star, c_elec_star, c_yzr_star):
    r = (-e * NA * ((c_vac_star * c_ref * charges["vac"]) + (c_elec_star * c_ref * charges["elec"]) + (c_yzr_star * c_ref * charges["yzr"])))
    return r

# not used when i_a = 0
def K(eta, x):
    r = integrate(Lambda_c(eta), x)
    return r

# Dimensionless:
def c_star(c):
    cs = c / c_ref
    return cs

d_vac_star = vac["diffusivity"] / d_ref
d_elec_star = elec["diffusivity"] / d_ref
d_yzr_star = yzr["diffusivity"] / d_ref

# def phi_star(phi):
#     ps = phi / phi_ref
#     return ps

def rho_star(c_vac_star, c_elec_star, c_yzr_star):
    rs = rho(c_vac_star, c_elec_star, c_yzr_star) / rho_ref
    return rs

t_star = t / t_ref
x_min_star = xmin / L_ref
x_max_star = xmax / L_ref
M_star = 1 # (M * t_ref) / (L_ref ** 2)
k_star = 4 # k / L_ref
lbd = (rho_ref * (L_ref ** 2)) / (phi_ref * epsilon_r * epsilon_0) # lambda
alp_vac = (d_vac_star * d_ref * t_ref) / (kB * T * (L_ref ** 2)) # alpha
alp_elec = (d_elec_star * d_ref * t_ref) / (kB * T * (L_ref ** 2))
alp_yzr = (d_yzr_star * d_ref * t_ref) / (kB * T * (L_ref ** 2))
bt_vac = (d_vac_star * d_ref * t_ref) / (L_ref ** 2) # Beta
bt_elec = (d_elec_star * d_ref * t_ref) / (L_ref ** 2)
bt_yzr = (d_yzr_star * d_ref * t_ref) / (L_ref ** 2)
sig_vac = alp_vac * e * phi_ref # sigma
sig_elec = alp_elec * e * phi_ref
sig_yzr = alp_yzr * e * phi_ref

# inputs
x_pred_val = np.linspace(-25, 25, 10000).reshape(-1, 1)
y_pred_val = np.ones_like(x_pred_val) * 1

inputs = {
    "x": torch.as_tensor(x_pred_val, dtype=torch.float32),
    "y": torch.as_tensor(y_pred_val, dtype=torch.float32),
}

class PDE_Function1(PDE):
    def __init__(self):
        # x_star, y_star
        x = Symbol("x")     # space
        y = Symbol("y")     # time

        input_variables = {"x": x, "y": y}
        eta = Function("eta")(*input_variables)

        #Cahn-Hilliard Eqs
        self.equations = {}
        self.equations["phase_variable_dx"] = (
            eta.diff(x, 1)
        )
        self.equations["Cahn-Hilliard"] = (
            eta.diff(y, 1) - ((4 * eta * (1 - eta) * (1 - 2 * eta)) - 4 * eta.diff(x, 2)).diff(x, 2)
        )

@physicsnemo.sym.main(config_path="conf", config_name="config_Adam_eta")
def run1(cfg: PhysicsNeMoConfig) -> None:
    pde1 = PDE_Function1()
    nr_layers = 4
    activation_eta = [Activation.TANH] * (nr_layers - 1) + [Activation.SIGMOID]
    fcn_cfg = dict(
        layer_size=32,
        nr_layers=4
    )
    FLC_eta = custom_FullyConnectedArch_eta(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("eta")],
        activation_fn=activation_eta,
        **fcn_cfg
    )
    nodes = (pde1.make_nodes() + [FLC_eta.make_node(name="FullyConnected_eta")])

    x, y = Symbol("x"), Symbol("y")
    rec = custom_Rectangle((-1, 0), (1, 1))

    PDE_domain = Domain()

    loss_ch = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"Cahn-Hilliard": 0},
        batch_size=cfg.batch_size.inter,
    )
    PDE_domain.add_constraint(loss_ch, "loss_ch")

    boundary_eta_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"eta": 0},
        batch_size= cfg.batch_size.initial,
        lambda_weighting= {"eta": 10},
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_eta_loss_left, "bc_eta_loss_left")

    boundary_eta_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"eta": 1},
        batch_size= cfg.batch_size.initial,
        lambda_weighting={"eta": 10},
        criteria=Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_eta_loss_right, "bc_eta_loss_right")

    boundary_eta_dx_loss = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phase_variable_dx": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, -1) | Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_eta_dx_loss, "boundary_eta_dx_loss")

    initial_eta_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"eta": 0},
        batch_size= cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x <=  0)
    )
    PDE_domain.add_constraint(initial_eta_loss_left, "initial_eta_loss_left")

    initial_eta_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"eta": 1},
        batch_size= cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x > 0)
    )
    PDE_domain.add_constraint(initial_eta_loss_right, "initial_eta_loss_right")

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=inputs,
        output_names=["eta"],
        batch_size=1024,
        plotter=etaCustomInferencerPlotter(),
    )
    PDE_domain.add_inferencer(grid_inference, "inf_data")

    slv = Solver(cfg, PDE_domain)
    slv.solve()

class PDE_Function(PDE):
    def __init__(self):
        # x_star, y_star
        x = Symbol("x")     # space
        y = Symbol("y")     # time

        input_variables = {"x": x, "y": y}
        eta = Function("eta")(*input_variables)
        phi_star = Function("phi_star")(*input_variables)
        c_vac_star = Function("c_vac_star")(*input_variables)
        c_elec_star = Function("c_elec_star")(*input_variables)
        c_yzr_star = Function("c_yzr_star")(*input_variables)

        self.equations = {}
        self.equations["flux_vac"] = (
            - ((d_vac_star * d_ref * c_vac_star * c_ref) / (kB * T * L_ref)) * mu_o_i(vac["mu_cathode"], vac["mu_YSZ"], eta).diff(x, 1) -
            (d_vac_star * d_ref * c_ref / L_ref) * c_vac_star.diff(x, 1) -
            (d_vac_star * d_ref * charges["vac"] * e * c_vac_star * c_ref * phi_ref) / (kB * T * L_ref) * phi_star.diff(x, 1)
        )
        self.equations["flux_elec"] = (
            - ((d_elec_star * d_ref * c_elec_star * c_ref) / (kB * T * L_ref)) * mu_o_i(elec["mu_cathode"], elec["mu_YSZ"], eta).diff(x, 1) -
            (d_elec_star * d_ref * c_ref / L_ref) * c_elec_star.diff(x, 1) -
            (d_elec_star * d_ref * charges["elec"] * e * c_elec_star * c_ref * phi_ref) / (kB * T * L_ref) * phi_star.diff(x, 1)
        )
        self.equations["flux_yzr"] = (
            - ((d_yzr_star * d_ref * c_yzr_star * c_ref) / (kB * T * L_ref)) * mu_o_i(yzr["mu_cathode"], yzr["mu_YSZ"], eta).diff(x, 1) -
            (d_yzr_star * d_ref * c_ref / L_ref) * c_yzr_star.diff(x, 1) -
            (d_yzr_star * d_ref * charges["yzr"] * e * c_yzr_star * c_ref * phi_ref) / (kB * T * L_ref) * phi_star.diff(x, 1)
        )
        self.equations["phi_dx"] = (
            phi_star.diff(x, 1)
        )
        #Poisson Eq
        self.equations["Poisson_eq"] = (
            phi_star.diff(x, 2) - (-lbd * rho_star(c_vac_star, c_elec_star, c_yzr_star))
        )
        # Concentration Eqs
        self.equations["conc_vac"] = (
            (c_vac_star.diff(y, 1) -
            (alp_vac * c_vac_star * mu_o_i(vac["mu_cathode"], vac["mu_YSZ"], eta).diff(x, 2) +
             bt_vac * c_vac_star.diff(x, 2) +
             sig_vac * c_vac_star * phi_star.diff(x, 2))
        ))
        self.equations["conc_elec"] = (
            (c_elec_star.diff(y, 1) -
            (alp_elec * c_elec_star * mu_o_i(elec["mu_cathode"], elec["mu_YSZ"], eta).diff(x, 2) +
             bt_elec * c_elec_star.diff(x, 2) +
             sig_elec * c_elec_star * phi_star.diff(x, 2))
        ))
        self.equations["conc_yzr"] = (
            (c_yzr_star.diff(y, 1) -
            (alp_yzr * c_yzr_star * mu_o_i(yzr["mu_cathode"], yzr["mu_YSZ"], eta).diff(x, 2) +
             bt_yzr * c_yzr_star.diff(x, 2) +
             sig_yzr * c_yzr_star * phi_star.diff(x, 2))
        ))

@physicsnemo.sym.main(config_path="conf", config_name="config_Adam")
def run2(cfg: PhysicsNeMoConfig) -> None:
    pde = PDE_Function()
    nr_layers = 4
    activation_conc = [Activation.TANH] * (nr_layers - 1) + [Activation.SOFTPLUS]
    activation_eta = [Activation.TANH] * (nr_layers - 1) + [Activation.SIGMOID]
    activation_phi = [Activation.TANH] * nr_layers
    fcn_cfg = dict(
        layer_size=32,
        nr_layers=4
    )
    FLC_eta = custom_FullyConnectedArch_eta(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("eta")],
        activation_fn=activation_eta,
        **fcn_cfg
    )
    FLC_phi = custom_FullyConnectedArch_phi(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("phi_star")],
        activation_fn=activation_phi,
        **fcn_cfg,
    )
    FLC_c_vac_star = custom_FullyConnectedArch_conc(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c_vac_star")],
        activation_fn=activation_conc,
        **fcn_cfg,
    )
    FLC_c_elec_star = custom_FullyConnectedArch_conc(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c_elec_star")],
        activation_fn=activation_conc,
        **fcn_cfg,
    )
    FLC_c_yzr_star = custom_FullyConnectedArch_conc(
        # y is time
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c_yzr_star")],
        activation_fn=activation_conc,
        **fcn_cfg,
    )

    for param in FLC_eta.parameters():
        param.requires_grad = False

    nodes = (pde.make_nodes() + [FLC_eta.make_node(name="FullyConnected_eta")] +
             [FLC_phi.make_node(name="FullyConnected_phi")] +
             [FLC_c_vac_star.make_node(name="FullyConnected_c_vac")] +
             [FLC_c_elec_star.make_node(name="FullyConnected_c_elec")] +
             [FLC_c_yzr_star.make_node(name="FullyConnected_c_yzr")])

    # class EvalMF(Module):
    #     def __init__(self, net):
    #         super().__init__()
    #         self.net = net.eval()
    #
    #     def forward(self, in_vars):  # Dict[str, Tensor]) -> Dict[str, Tensor]:
    #         out = self.net(in_vars)
    #         return torch.Tensor.detach(out)

    # evalmf_node = Node(["x", "y"], ["eta"], torch.Tensor.detach(FLC_eta))
    # nodes = nodes + [evalmf_node]

    x, y = Symbol("x"), Symbol("y")
    rec = custom_Rectangle((-1, 0), (1, 1))

    PDE_domain = Domain()

    loss_poisson = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"Poisson_eq": 0},
        batch_size=cfg.batch_size.inter,
    )
    PDE_domain.add_constraint(loss_poisson, "loss_poisson")

    loss_c_vac_eq = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"conc_vac": 0},
        batch_size=cfg.batch_size.inter,
    )
    PDE_domain.add_constraint(loss_c_vac_eq, "loss_c_vac_eq")

    loss_c_elec_eq = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"conc_elec": 0},
        batch_size=cfg.batch_size.inter,
    )
    PDE_domain.add_constraint(loss_c_elec_eq, "loss_c_elec_eq")

    loss_c_yzr_eq = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"conc_yzr": 0},
        batch_size=cfg.batch_size.inter,
    )
    PDE_domain.add_constraint(loss_c_yzr_eq, "loss_c_yzr_eq")

    boundary_flux_vac_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_vac": i_a / (e * NA * charges["vac"])},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_flux_vac_loss_right, "boundary_flux_vac_loss_right")

    boundary_flux_elec_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_elec": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_flux_elec_loss_right, "boundary_flux_elec_loss_right")

    boundary_flux_yzr_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_yzr": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_flux_yzr_loss_right, "boundary_flux_yzr_loss_right")

    boundary_flux_vac_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_vac": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_flux_vac_loss_left, "boundary_flux_vac_loss_left")

    boundary_flux_elec_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_elec": i_a / (e * NA * charges["elec"])},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_flux_elec_loss_left, "boundary_flux_elec_loss_left")

    boundary_flux_yzr_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"flux_yzr": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_flux_yzr_loss_left, "boundary_flux_yzr_loss_left")

    boundary_phi_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phi_star": 0.927}, # 0.102 / phi_ref
        batch_size= cfg.batch_size.boundary,
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_phi_loss_left, "boundary_phi_loss_left")

    boundary_phi_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phi_star": 0},
        batch_size= cfg.batch_size.boundary,
        criteria=Eq(x, 1)
    )
    PDE_domain.add_constraint(boundary_phi_loss_right, "boundary_phi_loss_right")

    boundary_phi_dx_loss = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phi_dx": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, -1)
    )
    PDE_domain.add_constraint(boundary_phi_dx_loss, "boundary_phi_dx_loss")

    initial_phi_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phi_star": 0.927},
        batch_size= cfg.batch_size.initial,
        lambda_weighting={"phi_star": 10},
        criteria=Eq(y, 0) & (x < 0)
    )
    PDE_domain.add_constraint(initial_phi_loss_left, "initial_phi_loss_left")

    initial_phi_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"phi_star": 0.0},
        lambda_weighting={"phi_star": 10},
        batch_size= cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x > 0)
    )
    PDE_domain.add_constraint(initial_phi_loss_right, "initial_phi_loss_right")

    initial_conc_vac_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_vac_star": 0.05},
        lambda_weighting={"c_vac_star": 10},
        batch_size= cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x < 0)
    )
    PDE_domain.add_constraint(initial_conc_vac_loss_left, "initial_conc_vac_loss_left")

    initial_conc_vac_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_vac_star": 0.5},
        lambda_weighting={"c_vac_star": 10},
        batch_size= cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x > 0)
    )
    PDE_domain.add_constraint(initial_conc_vac_loss_right, "initial_conc_vac_loss_right")

    initial_conc_elec_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_elec_star": 0.1},
        lambda_weighting={"c_elec_star": 10},
        batch_size=cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x < 0)
    )
    PDE_domain.add_constraint(initial_conc_elec_loss_left, "initial_conc_elec_loss_left")

    initial_conc_elec_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"c_elec_star": 0},
        lambda_weighting={"c_elec_star": 10},
        batch_size=cfg.batch_size.initial,
        fixed_dataset=False,
        criteria=Eq(y, 0) & (x > 0)
    )
    PDE_domain.add_constraint(initial_conc_elec_loss_right, "initial_conc_elec_loss_right")

    initial_conc_yzr_loss_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_yzr_star": 0},
        lambda_weighting={"c_yzr_star": 10},
        batch_size=cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x < 0)
    )
    PDE_domain.add_constraint(initial_conc_yzr_loss_left, "initial_conc_yzr_loss_left")

    initial_conc_yzr_loss_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_yzr_star": 1},
        lambda_weighting={"c_yzr_star": 10},
        batch_size=cfg.batch_size.initial,
        criteria=Eq(y, 0) & (x > 0)
    )
    PDE_domain.add_constraint(initial_conc_yzr_loss_right, "initial_conc_yzr_loss_right")

    grid_inference = PointwiseInferencer(
        nodes=nodes,
        invar=inputs,
        output_names=["eta", "phi_star", "c_vac_star", "c_elec_star", "c_yzr_star"],
        batch_size=1024,
        plotter=CustomInferencerPlotter(),
    )
    PDE_domain.add_inferencer(grid_inference, "inf_data")

    slv = Solver(cfg, PDE_domain)
    slv.solve()

if __name__ == "__main__":
    run1()
    run2()



# def plot():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x_pred_val1 = np.linspace(0, xmax, 5000).reshape(-1, 1)
#     y_pred_val1 = np.ones_like(x_pred_val) * 1
#     input = {
#     "x": torch.as_tensor(x_pred_val1, dtype=torch.float32).to(device),
#     "y": torch.as_tensor(y_pred_val1, dtype=torch.float32).to(device),
#     }
#
#     fcn_cfg = dict(
#         layer_size=64,
#         nr_layers=5
#     )
#
#     nr_layers = 5
#     activation_conc = [Activation.TANH] * (nr_layers - 1) + [Activation.SOFTPLUS]
#     activation_eta = [Activation.TANH] * (nr_layers - 1) + [Activation.SIGMOID]
#     activation_phi = [Activation.TANH] * nr_layers
#
#
#     model_eta = custom_FullyConnectedArch_eta(
#         # y is time
#         input_keys=[Key("x"), Key("y")],
#         output_keys=[Key("eta")],
#         activation_fn=activation_eta,
#         **fcn_cfg
#     )
#     model_eta.make_node("eta")
#
#     model_phi = custom_FullyConnectedArch_phi(
#         input_keys=[Key("x"), Key("y")],
#         output_keys=[Key("phi")],
#         activation_fn=activation_phi,
#         **fcn_cfg
#     )
#     model_phi.make_node("phi")
#
#     model_c_vac = custom_FullyConnectedArch_conc(
#         input_keys=[Key("x"), Key("y")],
#         output_keys=[Key("c_vac_star")],
#         activation_fn=activation_conc,
#         **fcn_cfg
#     )
#     model_c_vac.make_node("c_vac")
#
#     model_c_elec = custom_FullyConnectedArch_conc(
#         input_keys=[Key("x"), Key("y")],
#         output_keys=[Key("c_elec_star")],
#         activation_fn=activation_conc,
#         **fcn_cfg
#     )
#     model_c_elec.make_node("c_elec")
#
#     model_c_yzr = custom_FullyConnectedArch_conc(
#         input_keys=[Key("x"), Key("y")],
#         output_keys=[Key("c_yzr_star")],
#         activation_fn=activation_conc,
#         **fcn_cfg
#     )
#     model_c_yzr.make_node("c_yzr")
#
#     checkpoint_path_eta = "outputs/Adam/FullyConnected_eta.0.pth"
#     checkpoint_path_phi = "outputs/Adam/FullyConnected_phi.0.pth"
#     checkpoint_path_c_vac = "outputs/Adam/FullyConnected_c_vac.0.pth"
#     checkpoint_path_c_elec = "outputs/Adam/FullyConnected_c_elec.0.pth"
#     checkpoint_path_c_yzr = "outputs/Adam/FullyConnected_c_yzr.0.pth"
#     state_dict_eta = torch.load(checkpoint_path_eta, map_location=device)
#     state_dict_phi = torch.load(checkpoint_path_phi, map_location=device)
#     state_dict_c_vac = torch.load(checkpoint_path_c_vac, map_location=device)
#     state_dict_c_elec = torch.load(checkpoint_path_c_elec, map_location=device)
#     state_dict_c_yzr = torch.load(checkpoint_path_c_yzr, map_location=device)
#     model_eta.load_state_dict(state_dict_eta)
#     model_phi.load_state_dict(state_dict_phi)
#     model_c_vac.load_state_dict(state_dict_c_vac)
#     model_c_elec.load_state_dict(state_dict_c_elec)
#     model_c_yzr.load_state_dict(state_dict_c_yzr)
#     model_eta.to(device)
#     model_phi.to(device)
#     model_c_vac.to(device)
#     model_c_elec.to(device)
#     model_c_yzr.to(device)
#     model_eta.eval()
#     model_phi.eval()
#     model_c_vac.eval()
#     model_c_elec.eval()
#     model_c_yzr.eval()
#
#     with torch.no_grad():
#         eta_pred = model_eta(input)["eta"].cpu().numpy()
#         phi_pred = model_phi(input)["phi"].cpu().numpy()
#         c_vac_pred = model_c_vac(input)["c_vac"].cpu().numpy()
#         c_elec_pred = model_c_elec(input)["c_elec"].cpu().numpy()
#         c_yzr_pred = model_c_yzr(input)["c_yzr"].cpu().numpy()
#
#         # Calculate Charge Density rho
#     rho_ = rho(c_vac_pred, c_elec_pred, c_yzr_pred)
#
#     # Visualization
#     fig, axes = plt.subplots(3, 2, figsize=(12, 12))
#     fig.suptitle(f"Phase-Field & Electrochemical Profiles at t = {y_pred_val[0, 0]}s", fontsize=16)
#
#     plots = [
#         (eta_pred, "Phase Field (η)", "blue"),
#         (phi_pred, "Electric Potential (φ)", "red"),
#         (c_vac_pred, "Vacancy Conc. (c_vac)", "green"),
#         (c_elec_pred, "Electron Conc. (c_elec)", "orange"),
#         (c_yzr_pred, "YZR Conc. (c_yzr)", "purple"),
#         (rho_, "Charge Density (ρ)", "black")
#     ]
#
#     for ax, (data, title, color) in zip(axes.flatten(), plots):
#         ax.plot(x_pred_val, data, color=color, linewidth=2)
#         ax.set_title(title)
#         ax.set_xlabel("x (nm)")
#         ax.set_ylabel("Value")
#         ax.grid(True, linestyle='--', alpha=0.6)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


    # plot()


# [14:38:15] - [step:      68200] loss:  1.050e+27, time/iteration:  2.039e+02 ms
# [14:38:35] - [step:      68300] loss:  1.037e+27, time/iteration:  2.039e+02 ms
# [14:38:55] - [step:      68400] loss:  1.045e+27, time/iteration:  2.041e+02 ms
# [14:39:16] - [step:      68500] loss:  1.022e+27, time/iteration:  2.039e+02 ms
# [14:39:36] - [step:      68600] loss:  1.011e+27, time/iteration:  2.043e+02 ms
# [14:39:46] - [step:      68648] loss went to INFs/NaNs



    # initial_left = PointwiseBoundaryConstraint(
    #     # For cathode side's condition
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar= {"flux_vac": i_a / (charges["vac"] * e * NA), "flux_elec": 0, "flux_yzr": 0,
    #             "c_vac": c_vac_init_Cathode, "c_elec": c_elec_init_Cathode, "c_yzr": c_yzr_init_Cathode},
    #     batch_size = cfg.batch_size.initial,
    #     criteria = x < 23
    # )
    #
    # PDE_domain.add_constraint(initial_left, "initial_left_cond")
    #
    # initial_right = PointwiseBoundaryConstraint(
    #     # For YSZ side's condition
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar={"flux_vac": 0, "flux_elec": i_a / (charges["elec"] * e * NA), "flux_yzr": 0,
    #             "c_vac": c_vac_init_YSZ, "c_elec": c_elec_init_YSZ, "c_yzr": c_yzr_init_YSZ},
    #     batch_size=cfg.batch_size.initial,
    #     criteria = x > 27
    # )
    #
    # PDE_domain.add_constraint(initial_right, "initial_right_cond")
    #
    # boundary_left = PointwiseBoundaryConstraint(
    #     # Cathode side boundary condition
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar={"phi": 0.102, "eta": 0},
    #     batch_size=cfg.batch_size.initial,
    #     criteria = Eq(x, 0)
    # )
    #
    # PDE_domain.add_constraint(boundary_left, "boundary_left_cond")
    #
    # boundary_right = PointwiseBoundaryConstraint(
    #     # YSZ side boundary condition
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar={"phi": 0., "eta": 1},
    #     batch_size=cfg.batch_size.initial,
    #     criteria = Eq(x, xmax)
    # )
    #
    # PDE_domain.add_constraint(boundary_right, "boundary_right_cond")
    #
    # initial_inter = PointwiseBoundaryConstraint(
    #     # interface initial condition
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar={"c_vac": c_vac_init_YSZ, "c_elec": c_elec_init_YSZ, "c_yzr": c_yzr_init_YSZ},
    #     batch_size=cfg.batch_size.initial,
    #     criteria = Eq(y, 0) & (x >= 23) & (x <= 27)
    # )
    #
    # PDE_domain.add_constraint(initial_inter, "initial_inter_cond")
    #
    # inter = PointwiseInteriorConstraint(
    #     # interface pde functions
    #     nodes=nodes,
    #     geometry=rec,
    #     outvar={"Cahn-Hilliard": 0, "Poisson_eq": 0, "conc_vac": 0, "conc_elec": 0, "conc_yzr": 0},
    #     batch_size=cfg.batch_size.inter,
    #     criteria = (x >= 23) & (x <= 27)
    # )
    #
    # PDE_domain.add_constraint(inter, "inter_cond")

#
# self.equations["conc_vac"] = (
#         c_vac.diff(y, 1) -
#         (vac["diffusivity"] * c_vac) / (kB * T) * (mu(vac["mu_YSZ"], vac["mu_cathode"], eta)).diff(x, 2) +
#         vac["diffusivity"] * c_vac.diff(x, 2) +
#         (vac["diffusivity"] * charges["vac"] * e * c_vac) / (kB * T) * phi.diff(x, 2) -
#         vac["rate_constant"] * (Integral(A_c(eta), x) * A_c(eta))
# )
# self.equations["conc_elec"] = (
#         c_elec.diff(y, 1) -
#         (elec["diffusivity"] * c_elec) / (kB * T) * (mu(elec["mu_YSZ"], elec["mu_cathode"], eta)).diff(x, 2) +
#         elec["diffusivity"] * c_elec.diff(x, 2) +
#         (elec["diffusivity"] * charges["elec"] * e * c_elec) / (kB * T) * phi.diff(x, 2) -
#         elec["rate_constant"] * (Integral(A_c(eta), x) * A_c(eta))
# )
# self.equations["conc_yzr"] = (
#         c_yzr.diff(y, 1) -
#         (yzr["diffusivity"] * c_yzr) / (kB * T) * (mu(yzr["mu_YSZ"], yzr["mu_cathode"], eta)).diff(x, 2) +
#         yzr["diffusivity"] * c_yzr.diff(x, 2) +
#         (yzr["diffusivity"] * charges["yzr"] * e * c_yzr) / (kB * T) * phi.diff(x, 2) -
#         yzr["rate_constant"] * (Integral(A_c(eta), x) * A_c(eta))
# )