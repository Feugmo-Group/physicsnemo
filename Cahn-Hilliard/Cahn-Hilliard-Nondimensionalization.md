# Dimensionless Equations

> 1. $$Poisson\ Equation: \frac{\partial^2 \varphi^*}{\partial x^{*2}} = \frac{-\rho}{\varepsilon_r \varepsilon_0}$$
> 
> 2. $$Cahn\ Hilliard: \frac{\partial \eta}{\partial t^* } = M^* \frac{\partial^2 (4\eta(1-\eta)(1 - 2\eta) - k^*\frac{\partial^2 \eta}{\partial x^{*2}})}{\partial x^{*2}}$$
> 
> 3. $$Concentration_i:\frac{\partial c_i^* }{\partial t^* } = \alpha c_i^* \frac{\partial^2 \mu^o_i}{\partial x^{* 2}} + \beta\frac{\partial^2 c_i^* }{\partial x^{* 2}} + \sigma c_i^* \frac{\partial^2 \varphi^*}{\partial x^{*2}}-R^{3PB}_iK\Lambda_c$$
> 
> 4. $$J_i = -\frac{D_i^* D_{ref}c^* _ i c_{ref}}{k_BTL_{ref}}\frac{\partial \mu_i^o}{\partial x^* }-\frac{D^* _ iD_{ref}c_{ref}}{L_{ref}}\frac{\partial c^* _ i}{\partial x^* } - \frac{D^* _ iD_{ref}z_iec^* _ ic_{ref}}{k_BT}\frac{\varphi_{ref}}{L_{ref}}\frac{\partial \varphi^* }{\partial x^*}$$


# Nondimensionalization Parameters
|Original Parameter|Characteristic Value|Nondimensionalization|
|--------------|--------------------|---------------------|
|x|$$L_{ref}=2.5e^{-8}$$|$$x^*=\frac{x}{L_{ref}}$$|
|t|$$t_{ref} = \frac{(L_{ref})^2}{D_{ref}}: using\ 6e^{-8} \ for\ now$$|$$t^*=\frac{t}{t_{ref}}$$|
|$$D_i$$|$$D_{ref}=D_{vac}=1e^{-8}$$|$$D_i^*=\frac{D_i}{D_{ref}}$$|
|$$c_i$$|$$c_{ref}=1660$$|$$c_i^*=\frac{c_i}{c_{ref}}$$|
|$$\rho$$|$$\rho_{ref}=1.6e^6$$|$$\rho^*=\frac{\rho}{\rho_{ref}}$$|
|$$\varphi$$|$$\varphi_{ref}=\frac{k_B * T}{e}\approx0.11$$|$$\varphi^*=\frac{\varphi}{\varphi_{ref}}$$|
|$$M$$||$$M^* = \frac{Mt_{ref}}{(L_{ref})^2} = 1$$|
|$$k$$||$$k^* = \frac{k}{L_{ref}} = 4$$|

| | |
|-|-|
|$$\lambda$$|$$\frac{\rho_{ref}(L_{ref})^2}{\varphi_{ref}\varepsilon_r \varepsilon_0}$$|
|$$\alpha_i$$|$$\frac {D^* _ iD_{ref}t_{ref}}{k_BT(L_{ref})^2}$$|
|$$\beta_i$$|$$\frac {D_i^* D_{ref}t_{ref}}{(L_{ref})^2}$$|
|$$\sigma_i$$|$$\alpha_i \varphi_{ref}e$$|

# Physics Loss
> $$Cahn\ Hillard = 0$$
> $$Poisson\ Equation = 0$$
> $$Concentration_{vac} = 0$$
> $$Concentration_{elec} = 0$$
> $$Concentration_{yzr} = 0$$

# Boundary Condition 
#### $$x^* \in[-1, 1], t^* \in [0, 1]$$
|$$x^* = -1$$|$$x^* = 1$$|
|--------|-------|
|$$\eta = 0$$|$$\eta = 1$$|
|$$\frac{\partial {\eta}}{\partial x^*} = 0$$|$$\frac{\partial {\eta}}{\partial x^*} = 0$$|
|$$J_{vac} = 0$$|$$J_{vac} = \frac{i_a}{eN_Az_{vac}}$$|
|$$J_{elec} = \frac{i_a}{eN_Az_{elec}}$$|$$J_{elec} = 0$$| 
|$$J_{yzr} = 0$$|$$J_{yzr} = 0$$|
|$$\varphi^* = 0.927=\frac{0.102}{\varphi_{ref}}$$|$$\varphi_{ref} = 0$$|
|$$\frac{\partial \varphi^* }{\partial x^* } = 0$$| |

# Initial Condition ($$t^* = 0$$)
#### $$x^* \in[-1, 1], t^* \in [0, 1]$$
|$$x^* < 0$$|$$x^* > 0$$|
|--------|-------|
|$$\eta = 0$$|$$\eta = 1$$|
|$$\varphi^* = 0.927$$|$$\varphi^* = 0$$|
|$$c_{vac} = 0.05 = \frac{c_{vac\ bulk\ cathode}}{c_{ref}}$$|$$c_{vac} = 0.5 = \frac{c_{vac\ bulk\ YSZ}}{c_{ref}}$$|
|$$c_{elec} = 0.1 = \frac{c_{elec\ bulk\ cathode}}{c_{ref}}$$|$$c_{elec} = 0 = \frac{c_{elec\ bulk\ YSZ}}{c_{ref}}$$|
|$$c_{yzr} = 0 = \frac{c_{yzr\ bulk\ cathode}}{c_{ref}}$$|$$c_{yzr} = 1 = \frac{c_{yzr\ bulk\ YSZ}}{c_{ref}}$$|

# Outputs
- $$\eta$$
- $$\varphi^* $$
- $$c^* _{vac}$$
- $$c^* _{elec}$$
- $$c^* _{yzr}$$
- $$\rho^* $$