# Lithium Graphite Carbon Coating
Solving phase-field model with spherical coordinates (R). Left side is graphite, right side is carbon coating. Initially, Li only appear at the boundary of carbon coating.

# Equations
> 1. Allen Cahn:
> $$\frac{\partial \eta}{\partial t} = L[W_0(\frac{\partial^2 \eta}{\partial r^2} + \frac{2}{r}\frac{\partial \eta}{\partial r}) - \frac{\partial f(c_{Li},\eta)}{\partial \eta}]$$
> 
> 2. $$Concentration_{Li}$$:
> $$\frac{\partial c_{Li}}{\partial t} = \frac{D_{eff}c_{Li}}{k_BT}(\frac{\partial^2 \mu_o^i}{\partial r^2} + \frac{2}{r} \frac{\partial \mu_i^o}{\partial r}) + D_{eff}(\frac{\partial^2 c_{Li}}{\partial r^2} + \frac{2}{r} \frac{\partial c_{Li}}{\partial r}) + \frac{D_{eff}z_{Li}ec_{Li}}{k_BT}(\frac{\partial^2 \varphi}{\partial r^2} + \frac{2}{r} \frac{\partial \varphi}{\partial r}) - RK\Lambda_c$$
> 
> 3. Poisson Equation:
> $$\frac{\partial^2 \varphi}{\partial r^2} + \frac{2}{r}\frac{\partial \varphi}{\partial r} =  \frac{-\rho}{\varepsilon_0\varepsilon_r}$$
> 
> 4. Flux: 
> $$J_i = -\frac{D_{eff}c_i}{k_BT}\nabla\mu_i^o-D_{eff}\nabla c_i - \frac{D_{eff}z_iec_i}{k_BT}\nabla \varphi$$

# Functions
> 1. $$h(\eta):\eta^3(6\eta^2-15\eta+10)\ (\eta=0:graphite,1<\eta<1:interface, \eta = 1:carbon-coating)$$
> 
> 2. $$D_{eff} = D_{Graphite}h(\eta)+D_{carbon-coating}(1-h(\eta))$$
> 
> 3. $$f(c_{Li}, \eta): (c_{Li}-c_{Li}^0)^2h(\eta)+(c_{Li} - 0)^2(1 - h(\eta)) + 2(\eta^4 - 2\eta^3 + \eta^2)$$
> 
> 4. $$\Lambda_c = \eta^2(1 - \eta)^2$$
> 
> 5. $$\mu_i^o = \mu_{carbon-coating} + h(\eta)(\mu_{graphite} - \mu_{carbon-coating})$$
> 
> 6. $$\rho = -eN_Az_{Li}c_{Li}\ (-eN_Az_{Li}c_{Li}^* c_{ref})$$
> 
> 7. $$R_{ct} = \frac{RT}{Fi_0}$$
> 
> 8. $$K = \frac{1}{\int_{-\infty}^{\infty}\Lambda_cdx}$$ Not used when 0 $$i_a$$

# Constants Using
| constants | value | unit|
|-----------|-------|-----|
|$$k_B$$: Boltzmann constant|$$1.380649e^{-23}$$|$$JK^{-1}$$|
|$$e$$: Elementary charge|$$1.60217663e^{-19}$$|$$C$$|
|$$N_A$$: Avogadro's constant|$$6.02214076e^{23}$$|$$mol^{-1}$$|
|$$F$$: Faraday constant|$$9.648533e^4$$|$$C·mol^{-1}$$|
|$$r_{min} (radius)$$|0/0|$$m/nm$$|
|$$r_{max} (radius:\ graphite + carbon-coating)$$|$$8.505e^{-6}/8505$$|$$m/nm$$|
|$$t(y)$$: time |$$6e^{-8}/60$$|$$s/ns$$|
|$$\varepsilon_0$$|$$8.854187817e^{-12}$$|$$Fm^{-1}$$|
|$$\varepsilon_r$$|40| |
|$$T$$|293|$$K$$|

### Li:
 
  | graphite | carbon coating |
  |----------|----------------|
  |``D:6.4e^{-7}``|``D:1e^{-6}``|
  |<mark>``\mu_{graphite}:``</mark>|<mark>``\mu_{carbon-coating}:``</mark>|
  |``conc_{init}:0``|``conc_{init}:0``|
  
  - <mark>$$L(Mobility)：$$</mark>
  - $$charge(z): 1$$
  - $$ boundary_{carbon\ coating\ conc\ init}:1000\ mol / m^{-3}$$
  - $$rate\_constant(R): \mid \frac{i_a}{e N_A z_{Li}} \mid$$

# Outputs:
- $$c_{Li}$$
- $$\eta$$
- $$\varphi$$
- $$\rho$$

# Physics Loss
- $$Allen\ Cahn = 0$$
- $$concentration_{Li} = 0$$
- $$Poisson\ Equation = 0$$

### Verify ``i_0: 0.38-0.52\ mA·\ cm^2``