import numpy as np
from physicsnemo.sym.node import Node
from physicsnemo.sym.eq.pde import PDE
import physicsnemo
import sympy
from sympy import Symbol, Number, Function, Eq, Integral, integrate
from physicsnemo.sym.models.activation import Activation
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.key import Key
import torch
from scipy.constants import elementary_charge, Avogadro, k as Boltzmann, gas_constant
from geometry.custom_rec import custom_Rectangle
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from toy_model_custom_inferencer import Li_Graphite_CustomInferencerPlotter
import math
from models.custom_fullyconnected_eta import custom_FullyConnectedArch_eta
from models.custom_fullyconnected_c_li import custom_FullyConnectedArch_c_li
from models.custom_fullyconnected_mu import custom_FullyConnectedArch_mu

# constants set
L_g = 8.5-6 # m Single-particle probe ~17 micrometer diameter
T = 293 # K Paper section 2.4
c_max = 26500 # mol m^-3 Theoretical LiC6
omega_0 = -31861 # J mol^-1
A_0 = 2.5e7 # J m^-3 Energy barrier between Stage 1 and Stage 3
gamma = 1.2e7 # J m^-3
k_c = 1.2e-27 # J m^2 (mol m^-3)^-1
k_eta = 8e-10 # J m^-1
L_eta = 3e-19 # m^3 J^-1 s^-1 GITT pulse timing Figure 3e
V_m = 3.5e-5 # m^3 mol^-1 Graphite molar volume
D_Li = 6.4e-11 # m^2 s^-1 GITT Figure 3f caption
# i_0 = 0.285 # mA cm^-2
j_app = 6e-3 # Am^-2
tau = (L_g ** 2) / D_Li # time 1.1289s
R = gas_constant # gas constant
F = 96485.3321 # sA / mol Faraday constant
t = tau # s
L_ref = 8.5e-6
x_star = L_g / L_ref
t_star = t / tau

# inputs
x_pred_val = np.linspace(0, 1, 10000).reshape(-1, 1)
y_pred_val1 = np.ones_like(x_pred_val) * t * 0.3
y_pred_val2 = np.ones_like(x_pred_val) * t * 0.5
y_pred_val3 = np.ones_like(x_pred_val) * t * 0.7
y_pred_val_final = np.ones_like(x_pred_val) * t

input1 = {
    "x": torch.as_tensor(x_pred_val, dtype=torch.float32),
    "y": torch.as_tensor(y_pred_val1, dtype=torch.float32),
}
input2 = {
    "x": torch.as_tensor(x_pred_val, dtype=torch.float32),
    "y": torch.as_tensor(y_pred_val2, dtype=torch.float32),
}
input3 = {
    "x": torch.as_tensor(x_pred_val, dtype=torch.float32),
    "y": torch.as_tensor(y_pred_val3, dtype=torch.float32),
}
input_final = {
    "x": torch.as_tensor(x_pred_val, dtype=torch.float32),
    "y": torch.as_tensor(y_pred_val_final, dtype=torch.float32),
}


# functions
def M(c_li):
    r = (D_Li * c_max / (R * T)) * (c_li / c_max) * (1 - c_li / c_max)
    return r

#Dimensionless PDE equations
class GoverningPDE(PDE):
    def __init__(self):
        x = Symbol("x")     # space
        y = Symbol("y")     # time

        input_variables = {"x": x, "y": y}
        c_li = Function("c_li")(*input_variables)
        eta = Function("eta")(*input_variables)
        mu = Function("mu")(*input_variables)

        self.equations = {}
        self.equations["chemical-potential"] = (
            mu - ((R * T / V_m) * sympy.log(x * L_ref / (1 - x * L_ref)) +
             omega_0 / V_m * (1 - 2 * x * L_ref) - gamma / c_max * eta * (1 - eta)
             - k_c * c_li.diff(x, 2) * (1 / L_ref ** 2))
        )
        self.equations["Cahn-Hilliard"] = (
            c_li.diff(y, 1) - (tau / L_ref ** 2) * (M(c_li) * mu.diff(x, 1)).diff(x, 1)
        )
        self.equations["Allen-Cahn"] = (
            eta.diff(y, 1) + (L_eta * tau) * (2 * A_0 * eta * (1 - eta) * (1 - 2 * eta)
                                              - gamma * x * L_ref * (1 - 2 * eta)
                                              - k_eta * eta.diff(x, 2) * (1 / L_ref ** 2))
        )
        self.equations["dc_li_dx"] = (
            c_li.diff(x, 1)
        )
        self.equations["dmu_dx"] = (
            mu.diff(x, 1)
        )
        self.equations["deta_dx"] = (
            eta.diff(x, 1)
        )
        self.equations["j_app"] = (
            M(c_li) * mu.diff(x, 1) + (j_app * L_ref / F)
        )

@physicsnemo.sym.main(config_path="conf", config_name="config_Adam_toy_model")
def run(cfg: PhysicsNeMoConfig) -> None:
    pde = GoverningPDE()

    x, y = Symbol("x"), Symbol("y")
    rec = custom_Rectangle((0, 0), (x_star, t_star))
    pde_domain = Domain()

    FLC_eta = custom_FullyConnectedArch_eta(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("eta")],
    )
    FLC_c_li = custom_FullyConnectedArch_c_li(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("c_li")],
    )
    FlC_mu = custom_FullyConnectedArch_mu(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("mu")],
    )

    nodes = (pde.make_nodes() +
             [FLC_eta.make_node(name = "FLC_eta")] +
             [FLC_c_li.make_node(name = "FLC_c_li")] +
             [FlC_mu.make_node(name = "FlC_mu")])

    # PDE Loss ---------------------------------------------------
    loss_chemical_potential = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"chemical-potential": 0},
        batch_size=cfg.batch_size.interior,
    )
    pde_domain.add_constraint(loss_chemical_potential, "chemical-potential-loss")

    loss_cahn_hilliard = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"Cahn-Hilliard": 0},
        batch_size=cfg.batch_size.interior,
    )
    pde_domain.add_constraint(loss_cahn_hilliard, "Cahn-Hilliard-loss")

    loss_allen_cahn = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"Allen-Cahn": 0},
        batch_size=cfg.batch_size.interior,
    )
    pde_domain.add_constraint(loss_allen_cahn, "Allen-Cahn-loss")

    # Boundary Loss ---------------------------------------------------
    loss_boundary_c_li = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"dc_li_dx": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 0),
    )
    pde_domain.add_constraint(loss_boundary_c_li, "boundary_dc_li_dx_loss")

    loss_boundary_mu = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"dmu_dx": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 0),
    )
    pde_domain.add_constraint(loss_boundary_mu, "boundary_dmu_dx_loss")

    loss_boundary_eta = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"deta_dx": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, 0),
    )
    pde_domain.add_constraint(loss_boundary_eta, "boundary_deta_dx_loss")

    loss_boundary_j_app = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"j_app": 0},
        batch_size=cfg.batch_size.boundary,
        criteria=Eq(x, x_star),
    )
    pde_domain.add_constraint(loss_boundary_j_app, "boundary_j_app_loss")

    # Initial Loss ---------------------------------------------------
    loss_initial_fully_delithiated_c_li = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"c_li": 0.01 * c_max},
        batch_size=cfg.batch_size.initial,
        criteria=Eq(y, 0),
    )
    pde_domain.add_constraint(loss_initial_fully_delithiated_c_li, "initial_fully_delithiated_c_li_loss")

    loss_initial_fully_delithiated_eta = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=rec,
        fixed_dataset=False,
        outvar={"eta": 0},
        batch_size=cfg.batch_size.initial,
        criteria=Eq(y, 0),
    )
    pde_domain.add_constraint(loss_initial_fully_delithiated_eta, "initial_fully_delithiated_eta_loss")

    # Inferencer for 0.3t 0.5t 0.7t and full t
    grid_inference1 = PointwiseInferencer(
        nodes = nodes,
        invar = input1,
        output_names=["c_li", "eta", "mu"],
        batch_size=1024,
        plotter=Li_Graphite_CustomInferencerPlotter(),
    )
    pde_domain.add_inferencer(grid_inference1, "0.3t_inf_data")

    grid_inference2 = PointwiseInferencer(
        nodes=nodes,
        invar=input2,
        output_names=["c_li", "eta", "mu"],
        batch_size=1024,
        plotter=Li_Graphite_CustomInferencerPlotter(),
    )
    pde_domain.add_inferencer(grid_inference2, "0.5t_inf_data")

    grid_inference3 = PointwiseInferencer(
        nodes=nodes,
        invar=input3,
        output_names=["c_li", "eta", "mu"],
        batch_size=1024,
        plotter=Li_Graphite_CustomInferencerPlotter(),
    )
    pde_domain.add_inferencer(grid_inference3, "0.7t_inf_data")

    grid_inference4 = PointwiseInferencer(
        nodes=nodes,
        invar=input_final,
        output_names=["c_li", "eta", "mu"],
        batch_size=1024,
        plotter=Li_Graphite_CustomInferencerPlotter(),
    )
    pde_domain.add_inferencer(grid_inference4, "full_t_inf_data")

    slv = Solver(cfg, pde_domain)
    slv.solve()

if __name__ == "__main__":
    run()
