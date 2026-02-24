# Diffusion across cathode/electrolyte interface in SOFC
---
This model is using PINN(Physics Informed Neural Network) to solve an electrochemical phase-field model for Oxygen vacancy diffusion across cathode/electrolyte interface in solid oxide fuel cells.

---
# Main Equations
> 1. Poisson Equation:
> $$\nabla^2 \varphi = \frac{-\rho}{\varepsilon_r \varepsilon_0} $$
> 
> 2. Cahn Hilliard:
> $$\frac{\partial \eta}{\partial t} = M \nabla^2(4\eta(1-\eta)(1 - 2\eta)-k\nabla^2\eta)$$
> 
> 3. Concentration:
> $$\frac{\partial c_i}{\partial t} = \nabla \frac{D_ic_i}{k_BT}\nabla \mu^o_i+\nabla D_i\nabla c_i+\nabla \frac{D_iz_iec_i}{k_BT}\nabla \varphi-R^{3PB}_iK\Lambda_c = -\nabla J_i - R^{3PB}_iK\Lambda_c$$
> 
> 4. Flux:
> $$J_i = -\frac{D_ic_i}{k_BT}\nabla\mu_i^o-D_i\nabla c_i - \frac{D_iz_iec_i}{k_BT}\nabla \varphi$$

# Functions
> 1. $$\Lambda_c(\eta) = \eta^2 (1-\eta)^2$$
> 
> 2. $$h(\eta) = \eta^3  (6\eta^2-15\eta+10)$$
> 
> 3. $$\mu^o_i = \mu^{o,cathode}_i + (\mu^{o, YSZ}_i -\mu^{o,cathode}_i)h(\eta)$$
> 
> 4. $$\rho = -eN_A\sum c_iz_i\ \ \ \ (-eN_A\sum c_i^*c_{ref}z_i)$$
> 
> 5. $$K = \frac{1}{\int_{-\infty}^{\infty}\Lambda_cdx}$$ Not used when 0 $$i_a$$

# Contants
| Parameter | Value | Unit |
|-----------|-------|------|
|$$i_a$$|0|$$A·m^{-2}$$|
|$$k_B(Boltzmann\ constant)$$|$$1.380649e^{-23}$$|$$JK^{-1}$$|
|$$e(Elementary\ charge)$$|$$1.60217663e^{-19}$$|$$C$$|
|$$N_A(Avogadro's constant)$$|$$6.02214076e^{23}$$|$$mol^{-1}$$|
|$$T(Temperature)$$|$$1273$$|$$K$$|
|$$x_{min}$$|$$-2.5 * e^{-8}/-25$$|$$m/nm$$|
|$$x_{max}$$|$$2.5 * e^{-8}/25$$|$$m/nm$$|
|$$t(y)$$|$$6e^{-8}/60$$|$$s/ns$$|
|$$\varepsilon_r$$|40||
|$$\varepsilon_0$$|$$8.854187817e^{-12}$$|$$Fm^{-1}$$|

|$$element$$|$$charge(z)$$|$$\mu_{cathode}$$|$$\mu_{YSZ}$$|$$diffusivity(D)$$|$$Bulk_{cathode\ conc}(c)$$|$$Bulk_{YSZ\ conc}(c)$$|$$Rate\ Constant(R)$$|
|---------|--------|---|-------------|-------------|-----|----|---|
|vac|2.0|0.2e|0.12e|$$1.0e^{-8}$$|83.0|830.0|$$\mid \frac{i_a}{e * N_A * z_{vac}}\mid$$|
|elec|-1.0|0.0e|0.0e|$$2.0e^{-4}$$|166.0|0|$$\mid \frac{i_a}{e * N_A * z_{elec}}\mid$$|
|yzr|-1.0|0.1e|0.0e|$$5.0e^{-20}$$|0|1660.0|$$0$$|

# Outputs
- $$\eta$$ 
- $$\varphi$$
- $$c_{vac}$$
- $$c_{elec}$$
- $$c_{yzr}$$
- $$\rho$$

# Model Setting
#### 5 neural network for 5 outputs, $$\rho$$ is calculated using concentration
#### Train $$\eta$$ first with 10000 epochs then use trained $$\eta$$ as constant
> activation function($$\eta$$): Silu, Sigmoid(0-1)
> activation function($$c_i$$): Silu, Softplus(guarantee positive output)
> activation function($$\varphi$$): Tanh
#### Deep Fully Connected Neural Network
> optimizer: AdamW
> learning rate: $$1e^{-4}$$
> number of layers: 4
> layer size: 32
> NTK-based adaptive weighting
> Interior Sampling: 10000
> Boundary Sampling: 1500
> Total Epochs: 80000
> 
> Special Sampling method: 80% sampling point allocates at the interface.

#### see Cahn-Hilliard-Nondimensionalization for BC and IC
