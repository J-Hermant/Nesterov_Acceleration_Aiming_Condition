import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import PowerNorm
from tqdm import tqdm

from functions import g
from functions import grad_g
from functions import L_smooth
# Definition domain, grid
bound_square = 1.2638
x_vals = np.linspace(-bound_square, bound_square, 2000)
y_vals = np.linspace(-bound_square, bound_square, 2000)
X, Y = np.meshgrid(x_vals, y_vals)
epsilon = 0.1

Z = g(X, Y,epsilon)
# Gradient on the grid
U, V = grad_g(X, Y,epsilon)
norm_x2 = X**2 + Y**2
norm_grad2 = U**2 + V**2
inner_prod = U*X + V*Y
# Computation of L-smooth value
print("Computation of L-smooth value on the definition domain")
L = 0
for x in tqdm(x_vals):
    for y in y_vals:
        L=max(0,L_smooth(x, y,epsilon))
print("L = ", L)

# Compute the pointwise PL values on the grid
pl = np.zeros_like(X)
mask = Z > 1e-12
pl[mask] = norm_grad2[mask] / (2 * Z[mask])

print("Computation of convergence rates for strong quasar convexity")

gamma = np.linspace(1*10**-10,1*10**-2,1000)
gf_bound = np.empty(1000)
nmo_bound = np.empty(1000)
gd_bound = np.empty(1000)
nm_bound = np.empty(1000)
Q1 = np.zeros_like(X)
k = 0
for i in tqdm(gamma):
    Q1[mask] = -2*(Z[mask]- i**(-1)*inner_prod[mask]) / norm_x2[mask]
    gf_bound[k] = i*Q1.min()
    nmo_bound[k] = i*np.sqrt(Q1.min())
    gd_bound[k] = i*Q1.min()/L
    nm_bound[k] = i*np.sqrt(Q1.min()/L)
    k+=1
print(" Bound GD-PL : ", pl.min()/L, "Bound GD-SQC : ", np.nanmax(gd_bound), "Bound NM-SQC : ", np.nanmax(nm_bound))


outdir = Path(__file__).parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

### Plot 

fontsize = 15
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
norm = PowerNorm(gamma=0.6, vmin=Z.min(), vmax=Z.max()) 
ax1.view_init(elev=45, azim=-60)
ax1.plot_surface(X, Y, Z, cmap="hsv", alpha=0.8,antialiased=False,norm=norm)
ax1.set_zticks([])
ax1.set_xticks([-1,1])
ax1.set_yticks([-1,1])
ax1.set_xlabel(r'$x$',labelpad=-10)
ax1.set_ylabel(r'$y$',labelpad=-10)
ax1.set_zlabel(r'$f(x,y)$',labelpad=-15)
fig1.savefig(outdir /"surface_fx_y.png", dpi=300, bbox_inches="tight")