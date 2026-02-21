import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import Dict


import numpy as np
import scipy
import matplotlib.pyplot as plt

from typing import Dict
from scipy.constants import elementary_charge as e, Avogadro as NA, k as kB

class _Plotter:
    def __call__(self, *args):
        raise NotImplementedError

    def _add_figures(self, group, name, results_dir, writer, step, *args):
        "Try to make plots and write them to tensorboard summary"

        # catch exceptions on (possibly user-defined) __call__
        try:
            fs = self(*args)
        except Exception as e:
            print(f"error: {self}.__call__ raised an exception:", str(e))
        else:
            for f, tag in fs:
                f.savefig(
                    results_dir + name + "_" + tag + ".png",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                writer.add_figure(group + "/" + name + "/" + tag, f, step, close=True)
            plt.close("all")

    def _interpolate_2D(self, size, invar, *outvars):
        "Interpolate 2D outvar solutions onto a regular mesh"

        assert len(invar) == 2

        # define regular mesh to interpolate onto
        xs = [invar[k][:, 0] for k in invar]
        extent = (xs[0].min(), xs[0].max(), xs[1].min(), xs[1].max())
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], size),
            np.linspace(extent[2], extent[3], size),
            indexing="ij",
        )

        # interpolate outvars onto mesh
        outvars_interp = []
        for outvar in outvars:
            outvar_interp = {}
            for k in outvar:
                outvar_interp[k] = scipy.interpolate.griddata(
                    (xs[0], xs[1]), outvar[k][:, 0], tuple(xyi)
                )
            outvars_interp.append(outvar_interp)

        return [extent] + outvars_interp

T = 1273 # K Temp
charges = {"vac": 2.0, "elec": -1.0, "yzr": -1.0}
rho_ref = 1.6e6
c_ref = 1660
phi_ref = (kB * T) / e # V approx 0.11, 0.1906986527
def rho_(c_vac_star, c_elec_star, c_yzr_star):
    r = (-e * NA * ((c_vac_star * c_ref * charges["vac"]) + (c_elec_star * c_ref * charges["elec"]) + (c_yzr_star * c_ref * charges["yzr"])))
    return r

def rho_star(c_vac_star, c_elec_star, c_yzr_star):
    rs = rho_(c_vac_star, c_elec_star, c_yzr_star) / rho_ref
    return rs

def actual_phi(phi_star):
    return phi_star * phi_ref

class CustomInferencerPlotter(_Plotter):
    """
    Custom plotter to visualize Phase-Field & Electrochemical profiles
    in a 3x2 subplot grid.
    """

    def __call__(self, invar, outvar):
        epsilon_r = 40  # permittivity of free space
        epsilon_0 = 8.854187817e-12  # Vacuum permittivity [F/m]
        # Calculate Charge Density (rho)
        # We assume outvar contains c_vac, c_elec, and c_yzr
        rho = rho_star(outvar["c_vac_star"], outvar["c_elec_star"], outvar["c_yzr_star"])
        phi = actual_phi(outvar["phi_star"])
        # Prepare plot data based on your specific keys
        # Mapping: (key_in_outvar, Title, Color)
        # We use 5 slots since rho is removed; the 6th slot will be empty.
        plot_configs = [
            ("eta", "Phase Field (η)", "red"),
            ("phi_star", "Electric Potential (φ)", "yellow"),
            ("c_vac_star", "Vacancy Conc. (c_vac)", "black"),
            ("c_elec_star", "Electron Conc. (c_elec)", "blue"),
            ("c_yzr_star", "YZR Conc. (c_yzr)", "pink"),
            ("rho", "Charge Density (ρ)", "green")
        ]

        # 2. Create the 3x2 Subplot Figure
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))

        # Extract x and y (time) from invar
        # Assuming invar["x"] and invar["y"] are [N, 1] arrays
        x_val = invar["x"][:, 0]
        time_val = invar["y"][0, 0]

        axes_flat = axes.flatten()

        # 3. Loop through the variables and plot
        for i, (key, title, color) in enumerate(plot_configs):
            ax = axes_flat[i]

            # Extract data from outvar
            # outvar[key] usually has shape [N, 1], so we take [:, 0]
            data = rho[:, 0] if key == "rho" else outvar[key][:, 0]

            ax.plot(x_val, data, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)

        # 4. Handle the 6th (empty) subplot
        # axes_flat[5].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 5. Return a list of tuples: [(Figure, Tag)]
        # This structure is critical for the base class loop
        return [(fig, "profiles")]