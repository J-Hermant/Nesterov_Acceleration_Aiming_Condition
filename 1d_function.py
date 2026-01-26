import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator

from functions import f_1d 
from functions import f_1d_deriv
from functions import f_1d_hess 
from functions import f_quadratic_worst_case
from functions import f_pl_worst_case


# --- # Computation of geometrical parameters on 1d function # --- #


# --- # (I) Associated Figures: Figure 1 (left), Figure 2 # --- #

### Define domain
x_min = -2
x_max = 2

### Define grid
nb_step = 300000


tt = np.linspace(x_min, x_max, nb_step)

### Define parameters
a_1d = 0.19
b_1d = 5

### Compute values of constant of PL, QG^-, LS, and QG^+
pl = 0.5*f_1d_deriv(tt,a_1d,b_1d)**2/f_1d(tt,a_1d,b_1d)
qg =  2*f_1d(tt,a_1d,b_1d)/tt**2
pl_const = np.min(pl)
qg_upp = np.max(qg)
qg_inf = np.min(qg)
L_const = np.max(np.abs(f_1d_hess(tt, a_1d, b_1d)))

print(pl_const)
print("Start strongly quasar convex")
#---# Special case: test of quasar convex #---#

### grid for gamma. mu is nonpositive if allowing large values
# gamma = np.linspace(1*10**(-15),2.4*10**(-13),100)
n_gamma = 1000
gamma = np.linspace(1*10**(-15),1*10**(-1),n_gamma)

### Initializing variables which will be assigned the optimal rate of convergence, respectively for NM and for GD 
max_val_acc = 0
max_val_gd = 0

### L smooth values
L = np.abs(f_1d_hess(tt,a_1d,b_1d)).max()
### Initializing mu
vec_mu=np.empty(n_gamma)


k = 0
for i in gamma:
    ### for each gamma, compute at each point of the grid the largest admissible mu to have the strong quasar convex inequality 
    sqc_mu_cst = -2*(f_1d(tt,a_1d,b_1d) - i**(-1)*f_1d_deriv(tt,a_1d,b_1d)*tt)/(tt**2)
    
    # print("gamma = ", i, ", mu = ", sqc_mu_cst.min(), ", gamma sqrt mu/L = ",np.sqrt(sqc_mu_cst).min()*i/np.sqrt(L), ", gamma mu/L = ", (sqc_mu_cst).min()*i/L)
    max_val_acc = np.nanmax([np.sqrt(sqc_mu_cst).min()*i/np.sqrt(L),max_val_acc])
    max_val_gd = np.nanmax([(sqc_mu_cst).min()*i/L,max_val_gd])
    vec_mu[k] = sqc_mu_cst.min()
    k+=1
    if k%100==0:
        print(k)

print("For 1d case, the optimal convergence rates are for GD-PL", r"$mu/L =$", pl_const/L, "for GD-SQC" , r"$tau \mu/L = $", max_val_gd,"for NM-SQC",  r"$tau \sqrt{mu/L} =$", max_val_acc)


### -- PLOT -- ###
outdir = Path(__file__).parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

fontsize = 15
plt.figure(figsize=(4, 4))
functions_values = f_1d(tt,a_1d,b_1d)
plt.plot(tt, functions_values, lw=2, color="grey")
plt.xticks([])
plt.yticks([])
plt.ylabel(r"$f(t)$",fontsize = fontsize)
plt.xlabel("t",fontsize = fontsize)
plt.savefig(outdir / "f_1d_graph.pdf", dpi=300, bbox_inches="tight")


### -- Figure quasar convex -- ###
fontsize = 14

plt.figure(figsize=(8,3))
plt.plot(gamma[0:],(vec_mu*gamma/L)[0:],color ="b",label="GD",lw=2)
plt.plot(gamma[0:],(np.sqrt(vec_mu)*gamma/np.sqrt(L))[0:],color="r", label ="NM",lw=2)

plt.xlabel(r'$\tau$',labelpad=-10,fontsize=fontsize)
plt.ylabel("convergence rate",fontsize=fontsize)
plt.xticks([0,0.1])
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.legend(edgecolor=(1,1,1),fontsize=fontsize-1,loc="upper right")
plt.savefig(outdir / "sqc_bound.pdf", dpi=300, bbox_inches="tight")

plt.figure(figsize=(8,3))
plt.plot(gamma[0:],(vec_mu*gamma)[0:],color ="b",label="GF",lw=2)
plt.plot(gamma[0:],(np.sqrt(vec_mu)*gamma)[0:],color="r", label ="NMO",lw=2)

plt.xlabel(r'$\tau$',labelpad=-10,fontsize=fontsize)
plt.ylabel("convergence rate",fontsize=fontsize)
plt.xticks([0,2.4*10**(-13)])
plt.legend(edgecolor=(1,1,1),fontsize=fontsize-1)
plt.savefig(outdir / "sqc_bound_cont.pdf", dpi=300, bbox_inches="tight")




# --- # (II) Code to generate Figure 5 # --- #





x_min = -10
x_max = 10
nb_step = 1000000
tt = np.linspace(x_min, x_max, nb_step)
a_1d = 0.07
b_1d = 13
L = np.max(np.abs(f_1d_hess(tt, a_1d, b_1d)))
pl = 0.5*f_1d_deriv(tt,a_1d,b_1d)**2/f_1d(tt,a_1d,b_1d)
rsi = np.abs(f_1d_deriv(tt,a_1d,b_1d)/(tt))
qg =  2*f_1d(tt,a_1d,b_1d)/tt**2
pl_const = np.min(pl)
rsi_const  = np.min(rsi)
qg_upp = np.max(qg)
qg_inf = np.min(qg)
arg_inf = np.argmin(rsi_const)
print("Nest : ", (qg_inf/qg_upp)**(0.25) * np.sqrt(np.min(pl_const)/L), "GD : ", np.sqrt(np.min(pl_const)*qg_inf)/L)
print(((qg_inf/qg_upp)**(0.25) * np.sqrt(np.min(pl_const)/L))/(np.sqrt(np.min(pl_const)*qg_inf)/L))


fontsize = 14
plt.figure(figsize=(8,3))
plt.plot(tt, f_1d(tt, a_1d, b_1d), lw=2, color="grey",label=r"$y = f(t)$")
plt.plot(tt[200000:800000],0.5*qg_upp*tt[200000:800000]**2, color="blue",label = r'$y= \frac{L_0}{2} t^2$',lw=2)
plt.plot(tt,0.5*qg_inf*tt**2,label = r'$y = \frac{\mu_0}{2} t^2$', color="red",lw=2)
plt.xlabel('t',labelpad=0,fontsize=fontsize)
plt.ylabel(r"$y$",fontsize=fontsize)
plt.xticks([])
plt.yticks([])
plt.legend(edgecolor=(1,1,1),fontsize=fontsize)
plt.savefig(outdir / "quad_bound_diff.pdf", dpi=300, bbox_inches="tight")



# --- # (III) Code to generate Figure 6 # --- #


tt_minus = np.linspace(-1,0,1000)
tt_plus = np.linspace(0,1,1000)

plt.figure(figsize=(8,3))
fontsize = 14
fontsize_ticks = 18
plt.plot(tt_minus,f_quadratic_worst_case(tt_minus,0.1,2),lw=2,color="r",label="low curvature")
plt.plot(tt_plus,f_quadratic_worst_case(tt_plus,0.1,2),lw=2,color="b",label ="high curvature")
plt.xticks([0],[r"$t^\ast$"],fontsize=fontsize_ticks)
plt.yticks([])
plt.ylabel(r"$f(t)$",labelpad=0,fontsize=fontsize_ticks)
x_low = -0.2
y_low = f_quadratic_worst_case(np.array([x_low]), 0.1, 2)[0]
plt.annotate(
    r'Optimal $\mu$',
    xy=(x_low, y_low),
    xytext=(0, 0.5),
    textcoords='data',
    arrowprops=dict(
        arrowstyle='->',
        lw=1.5,
        shrinkA=0, shrinkB=5,
        connectionstyle="arc3,rad=-0.1"
    ),
    fontsize=fontsize,
    ha='right', va='center'
)

plt.legend(edgecolor=(1,1,1),fontsize=fontsize)
plt.savefig(outdir / "worst_case_quad.pdf", dpi=300, bbox_inches="tight")




plt.figure(figsize=(8,3))
plt.plot(tt_minus,f_pl_worst_case(tt_minus,0.1,1,2),lw=2,color="r",label="low curvature")
plt.plot(tt_plus,f_pl_worst_case(tt_plus,0.1,0.38,5),lw=2,color="b",label ="zone with plateau")
plt.xticks([0],[r"$t^\ast$"],fontsize=fontsize_ticks)
plt.yticks([])
plt.ylabel(r"$f(t)$",labelpad=0,fontsize=fontsize_ticks)

x_low = -0.2
y_low = f_quadratic_worst_case(np.array([x_low]), 0.1, 2)[0]


x_high = 0.7
y_high = f_pl_worst_case(np.array([x_high]), 0.1,0.38, 5)[0]

text_x = 0.25
text_y = 1.3

plt.text(text_x, text_y, r"Optimal $\mu$ ?",
         fontsize=fontsize, ha='center', va='center')


plt.annotate(
    "", 
    xy=(x_low, y_low),
    xytext=(text_x-0.225, text_y-0.1),
    arrowprops=dict(
        arrowstyle='->',
        lw=1.5,
        shrinkA=2, shrinkB=5,
        connectionstyle="arc3,rad=0.2"
    )
)


plt.annotate(
    "",
    xy=(x_high, y_high),
    xytext=(text_x+0.25, text_y-0.05),
    arrowprops=dict(
        arrowstyle='->',
        lw=1.5,
        shrinkA=2, shrinkB=5,
        connectionstyle="arc3,rad=-0.2"
    )
)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize-1.5,loc = "upper left")
plt.savefig(outdir / "worst_case_pl.pdf", dpi=300, bbox_inches="tight")