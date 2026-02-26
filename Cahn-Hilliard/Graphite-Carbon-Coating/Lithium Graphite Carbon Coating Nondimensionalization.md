# Dimensionless Equations
> 1. Allen Cahn:
> $$\frac{\partial \eta}{\partial t^* } = L^* W_0^* \frac{\partial^2 \eta}{\partial r^{* 2}} + \frac{2L^* W_0^* }{r^* }\frac{\partial \eta}{\partial r^* } - L^* \frac{\partial f^* }{\partial \eta}$$
> 
> 2. $$Concentration_{Li}:$$
> $$\frac{\partial c^ * }{\partial t^ * } = (\frac{D^ * c^ * }{k_BT}\frac{\partial^2 \mu_i^o}{\partial r^{ * 2}} + \frac{2D^ * c^ * }{r^ * }\frac{\partial \mu_i^o}{\partial r^ * }) + (D^ * \frac{\partial^2 c^ * }{\partial r^{ * 2}} + \frac{2D^ * }{r^ * }\frac{\partial c^ * }{\partial r^ * }) + (D^ * c^ * z_{Li}\frac{\partial^2 {\varphi^ * }}{\partial r^ { * 2}} + \frac{2D ^ * z_{Li}c^ * }{r^ * }\frac{\varphi^ * }{\partial r^ * }) - RK\Lambda_c$$
> 
> 3. Poisson Equation:
> $$\frac{\partial^2 \varphi^ * }{\partial r^ { * 2}} + \frac{2}{r^ * }\frac{\partial \varphi^ * }{\partial r^ *} = -\lambda\rho^ *$$
> 
> 4. Flux:
> $$J_i = -\frac{D^ * D_{ref}c^ * c_{ref}}{k_BTL_{ref}}\frac{\partial \mu_i^o}{\partial r^ * } - \frac{D^ * D_{ref}c_{ref}}{L_{ref}}\frac{\partial c^ * }{\partial r^ * } - \frac{D^ * D_{ref}z_ic^ * c_{ref}}{L_{ref}}\frac{\partial \varphi^ * }{\partial r^ *}$$

# Nondimensionalization Parameters
|Original Parameter|Characteristic Value|Nondimensionalization|
|--------------|--------------------|---------------------|
|r|$$L_{ref}=8.505e^{-6}$$|$$r^* =\frac{r}{L_{ref}}$$|
|t|$$t_{ref} = \frac{(L_{ref})^2}{D_{ref}}: using\ 6e^{-8}$$|$$t^* =\frac{t}{t_{ref}} = \frac{tD_{ref}}{(L_{ref})^2}$$|
|$$D$$|$$D_{ref}=D_{carbon-coating}=1e^{-6}$$|$$D^*=\frac{D}{D_{ref}}$$|
|$$c$$|$$c_{ref}=1000$$|$$c^* =\frac{c}{c_{ref}}$$|
|$$\rho$$|$$\rho_{ref}=1.6e^6$$|$$\rho^*=\frac{\rho}{\rho_{ref}}$$|
|$$\varphi$$|$$\varphi_{ref}=\frac{k_B * T}{e}\approx0.11$$|$$\varphi^*=\frac{\varphi}{\varphi_{ref}}$$|
|$$f$$|$$f_c = (1000)^2$$|$$f^* = \frac{f}{f_c}$$|
|$$L(Mobility)$$||$$L^* = \frac{Lf_c(L_{ref})^2}{D_{ref}}$$|
|$$W_0$$||$$W_0^* = \frac{W_0}{f_c(L_{ref})^2}$$|
|$$k$$||$$k^* = \frac{k}{L_{ref}}$$|

| | |
|-|-|
|$$\lambda$$|$$\frac{\rho_{ref}(L_{ref})^2}{\varepsilon_r \varepsilon_0 \varphi_{ref}}$$|

# Physics Loss
- $$Allen\ Cahn = 0$$
- $$concentration_{Li} = 0$$
- $$Poisson\ Equation = 0$$

# Boundary Condition
#### $$x^* \in[-1, 1], t^* \in [0, 1]$$
|$$x^* = -1$$|$$x^* = 1$$|
|--------|-------|
|$$\eta = 0$$|$$\eta = 1$$|
|$$\frac{\partial {\eta}}{\partial x^*} = 0$$|$$\frac{\partial {\eta}}{\partial x^*} = 0$$|


# Initial Condition ($$t^ * = 0$$)
#### $$x^* \in[-1, 1], t^* \in [0, 1]$$
|$$x^* < 0$$|$$x^* > 0$$|
|--------|-------|
|$$\eta = 0$$|$$\eta = 1$$|
|$$c_{Li} = 0.0 = \frac{c_{Li\ bulk\ cathode}}{c_{ref}}$$|$$c_{Li} = 0.0 = \frac{c_{Li\ bulk\ YSZ}}{c_{ref}}$$ (for 0 < x < 1)|
||$$c_{Li} = 1 = \frac{c_{Li\ boundary\ carbon\ coating\ init}}{c_{ref}}$$ (for x = 1)|


# Outputs
- $$c^ * _{Li}$$
- $$\eta$$
- $$\varphi^ * $$
- $$\rho^ * $$


