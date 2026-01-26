import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


from functions import F
from functions import grad_F
from functions import eigen_value
from functions import gd
from functions import nag

outdir = Path(__file__).parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

### L_smooth
epsilon = 0.001
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.linspace(-3, 3, 1000)
L = 0
for i in x:
    for j in y:
        val = eigen_value(i, j, epsilon)
        L = max(L,val)


### Code to generate Figure 4

n_iter = 1*10**3
init =  np.array([0,3])
x_gd,f_gd = gd(init,n_iter,1/L,epsilon)
x_nag_0,f_nag_0,corr_0 = nag(init,n_iter,1/L,0.6,epsilon)
x_nag_1,f_nag_1,corr_1 = nag(init,n_iter,1/L,0.7,epsilon)
x_nag_2,f_nag_2,corr_2 = nag(init,n_iter,1/L,0.8,epsilon)
fontsize = 14
fontsize_legend = 25
plt.figure(figsize=(6,6))

x_min,x_max = -1, 5.5
y_min, y_max = -2, 3.5

x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)

Z = F(X, Y,epsilon)

fontsize_legend = 16
plt.contourf(X, Y, np.log(Z), levels=10, cmap="viridis")



def plot_path(traj, label, lw=2.0, mark_every=(0, 1, 2, 3, 4, 5), zorder=3,color="b",size = 40):
    plt.plot(traj[0, :], traj[1, :], linewidth=lw, label=label, zorder=zorder,color=color)
    # T = traj.shape[1]
#     idx = []
#     for k in mark_every:
#         kk = (T + k) if k < 0 else k
#         if 0 <= kk < T:
#             idx.append(kk)
#     idx = sorted(set(idx))
    # ax.scatter(traj[0, idx], traj[1, idx], s=size, marker="8", zorder=zorder+1,color=color)


plt.scatter(x_gd[0, 0], x_gd[1, 0],color="black",label="Initialization",s=200)
plot_path( x_gd, "GD", lw=8,color="blue",size=100)
plot_path( x_nag_0, r"NM' $\alpha=0.6$", lw=6.0,color="orange",size=80)
plot_path( x_nag_1, r"NM' $\alpha=0.7$", lw=4,color="purple",size=60)
plot_path( x_nag_2, r"NM' $\alpha=0.8$", lw=2,color="pink",size=40)

# plt.scatter(x_gd[0, :], x_gd[1, :], s=100,c="blue", label="GD",marker="x")
# plt.scatter(x_nag_0[0, :], x_nag_0[1, :], s=80, label=r"NM' $\alpha = 0.6$", c='orange',marker="x")
# plt.scatter(x_nag_1[0, :], x_nag_1[1, :], s=60, label=r"NM' $\alpha = 0.7$", c='purple',marker="x")
# plt.scatter(x_nag_2[0, :], x_nag_2[1, :], s=40, label=r"NM' $\alpha = 0.8$", c='pink',marker="x")
# plt.scatter(x_gd[0, 0], x_gd[1, 0],color="black",label="Initialization")
plt.scatter(0,0,color="red",marker = "*",label = "Unique Minimizer",s=200)
plt.xlabel("x",fontsize =fontsize,labelpad=-15)
plt.xticks([x_min,x_max],fontsize =fontsize)
plt.yticks([y_min,y_max],fontsize =fontsize)
plt.ylabel("y",fontsize =fontsize,labelpad=-20)
plt.legend(fontsize=fontsize_legend)
plt.tight_layout()
plt.savefig(outdir / "toy_example_traj.pdf", dpi=300, bbox_inches="tight")


fig = plt.figure(figsize=(6,3))
fontsize_legend = 16
plt.plot(np.log(f_gd), label="GD",c="blue",lw=2)
plt.plot(np.log(f_nag_0), label=r"NM' $\alpha = 0.6$", c='orange',lw=2)
plt.plot(np.log(f_nag_1),label=r"NM' $\alpha = 0.7$", c='purple',lw=2)
plt.plot(np.log(f_nag_2), label=r"NM' $\alpha = 0.8$", c='pink',lw=2)
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((-8,2),fontsize =fontsize)
plt.ylabel("log(f)",fontsize =fontsize,labelpad=-20)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize_legend)
plt.tight_layout()

plt.savefig(outdir / "toy_example_fonc.pdf", dpi=300, bbox_inches="tight")



Bound_x = 2*np.pi
Bound_y = 10
x = np.linspace(-Bound_x,Bound_x, 400)
y = np.linspace(-Bound_y, Bound_y, 400)
epsilon = 0.001
X, Y = np.meshgrid(x, y)
Z = F(X, Y,epsilon)
fig = plt.figure(figsize=(6,6))
plt.contourf(X, Y, np.log(Z), cmap='viridis', edgecolor='none')
cbar = plt.colorbar(ticks=[np.min(np.log(Z)), np.log(Z).max()])
cbar.ax.set_yticklabels(["0", f"{np.max(np.log(Z)):.2f}"])
cbar.set_label(r"$f$",fontsize = fontsize,labelpad=-15)
x = np.linspace(-Bound_x, Bound_x, 20)
y = np.linspace(-Bound_y, Bound_y, 20)
X, Y = np.meshgrid(x, y)
U,V = grad_F(X, Y,epsilon)

# Z = F(X, Y,epsilon)
plt.quiver(X, Y, -U,-V, angles="xy", color="blue",width = 0.005)


plt.scatter(0,0,color="r",marker = "+",label = "Unique Minimizer")
plt.xticks([-Bound_x,Bound_x],fontsize =fontsize)
plt.yticks([-Bound_y,Bound_y],fontsize =fontsize)
plt.xlabel("x",fontsize =fontsize,labelpad=-15)
plt.ylabel("y",fontsize =fontsize,labelpad=-20)
arrow_proxy = Line2D(
    [0], [0],
    linestyle="None",                 
    marker=r"$\rightarrow$",         
    markersize=10,
    color="blue",
)
a = plt.scatter(0,0,color="red",marker = "*",label = "Unique Minimizer",s=200)
plt.legend([a,arrow_proxy], ["Unique Minimizer","Descent Direction"],fontsize=fontsize_legend,loc = "upper right")
plt.savefig(outdir / "toy_example_fig.pdf", dpi=300, bbox_inches="tight")


fig = plt.figure(figsize=(6,3))
fontsize_legend = 16
plt.plot(corr_0[0:n_iter], label=r"NM' $\alpha = 0.6$", c='orange',lw=2)
plt.plot(corr_1[0:n_iter],label=r"NM' $\alpha = 0.7$", c='purple',lw=2)
plt.plot(corr_2[0:n_iter], label=r"NM' $\alpha = 0.8$", c='pink',lw=2)
xx = np.arange(n_iter)
plt.plot(xx,np.zeros(n_iter)*xx,linestyle="--",color="black",lw=2)
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((-0.8,0,1),fontsize =fontsize)
plt.ylabel("Aiming condition",fontsize =fontsize,labelpad=-10)
ax = plt.gca() 
axins = inset_axes(ax, width="35%", height="60%", loc='lower center')

### Zoom 100 first iterations
axins.plot(corr_0[:100],  c='orange',  lw=2)
axins.plot(corr_1[:100],  c='purple', lw=2)
axins.plot(corr_2[:100],  c='pink',  lw=2)
axins.plot(xx[:100],(np.zeros(n_iter)*xx)[:100],linestyle="--",color="black",lw=2)

axins.set_xlim(-5, 100)
axins.set_ylim(min(corr_0[:100].min(), corr_1[:100].min(), corr_2[:100].min())-0.1,
               max(corr_0[:100].max(), corr_1[:100].max(), corr_2[:100].max())+0.1)

# Retirer les labels du zoom pour éviter le fouillis
axins.set_xticks([])
axins.set_yticks([])

# Petits traits reliant la fenêtre au graphe principal
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.4")
plt.tight_layout()
plt.savefig(outdir /"toy_example_corr.pdf", dpi=300, bbox_inches="tight")









### Code to generate Figure 8

n_iter = 3*10**3
init =  np.array([0,3])
x_gd,f_gd = gd(init,n_iter,1/L,epsilon)
x_nag_0,f_nag_0,corr_0 = nag(init,n_iter,1/L,0.6,epsilon)
x_nag_1,f_nag_1,corr_1 = nag(init,n_iter,1/L,0.7,epsilon)
x_nag_2,f_nag_2,corr_2 = nag(init,n_iter,1/L,0.8,epsilon)
fontsize = 14
fontsize_legend = 25
plt.figure(figsize=(6,6))

x_min,x_max = -1, 5.5
y_min, y_max = -2, 3.5

x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)

Z = F(X, Y,epsilon)

fontsize_legend = 16
plt.contourf(X, Y, np.log(Z), levels=10, cmap="viridis")



plt.scatter(x_gd[0, 0], x_gd[1, 0],color="black",label="Initialization",s=200)
plot_path( x_gd, "GD", lw=8,color="blue",size=100)
plot_path( x_nag_0, r"NM' $\alpha=0.6$", lw=6.0,color="orange",size=80)
plot_path( x_nag_1, r"NM' $\alpha=0.7$", lw=4,color="purple",size=60)
plot_path( x_nag_2, r"NM' $\alpha=0.8$", lw=2,color="pink",size=40)

# plt.scatter(x_nag_2[0, :], x_nag_2[1, :], s=40, label=r"NM' $\alpha = 0.8$", c='pink',marker="x")
# plt.scatter(x_nag_1[0, :], x_nag_1[1, :], s=60, label=r"NM' $\alpha = 0.7$", c='purple',marker="x")
# plt.scatter(x_nag_0[0, :], x_nag_0[1, :], s=80, label=r"NM' $\alpha = 0.6$", c='orange',marker="x")
# plt.scatter(x_gd[0, :], x_gd[1, :], s=100,c="blue", label="GD",marker="x")

plt.scatter(0,0,color="red",marker = "*",label = "Unique Minimizer",s=200)
plt.xlabel("x",fontsize =fontsize,labelpad=-15)
plt.xticks([x_min,x_max],fontsize =fontsize)
plt.yticks([y_min,y_max],fontsize =fontsize)
plt.ylabel("y",fontsize =fontsize,labelpad=-20)
plt.legend(fontsize=fontsize_legend)
plt.tight_layout()
plt.savefig(outdir / "toy_example_traj_long.pdf", dpi=300, bbox_inches="tight")


fig = plt.figure(figsize=(6,3))
fontsize_legend = 16
plt.plot(np.log(f_gd), label="GD",c="blue",lw=2)
plt.plot(np.log(f_nag_0), label=r"NM' $\alpha = 0.6$", c='orange',lw=2)
plt.plot(np.log(f_nag_1),label=r"NM' $\alpha = 0.7$", c='purple',lw=2)
plt.plot(np.log(f_nag_2), label=r"NM' $\alpha = 0.8$", c='pink',lw=2)
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((-10,2),fontsize =fontsize)
plt.ylabel("log(f)",fontsize =fontsize,labelpad=-20)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize_legend)
plt.tight_layout()

plt.savefig(outdir / "toy_example_fonc_long.pdf", dpi=300, bbox_inches="tight")



fig = plt.figure(figsize=(6,3))
fontsize_legend = 16
plt.plot(corr_0[0:n_iter], label=r"NM' $\alpha = 0.6$", c='orange',lw=2)
plt.plot(corr_1[0:n_iter],label=r"NM' $\alpha = 0.7$", c='purple',lw=2)
plt.plot(corr_2[0:n_iter], label=r"NM' $\alpha = 0.8$", c='pink',lw=2)
xx = np.arange(n_iter)
plt.plot(xx,np.zeros(n_iter)*xx,linestyle="--",color="black",lw=2)
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((-0.8,0,1),fontsize =fontsize)
plt.ylabel("Aiming condition",fontsize =fontsize,labelpad=-10)
ax = plt.gca() 
axins = inset_axes(ax, width="35%", height="60%", loc='lower center')

### Zoom 100 first iterations
axins.plot(corr_0[:100],  c='red',  lw=2)
axins.plot(corr_1[:100],  c='green', lw=2)
axins.plot(corr_2[:100],  c='pink',  lw=2)
axins.plot(xx[:100],(np.zeros(n_iter)*xx)[:100],linestyle="--",color="black",lw=2)

axins.set_xlim(-5, 100)
axins.set_ylim(min(corr_0[:100].min(), corr_1[:100].min(), corr_2[:100].min())-0.1,
               max(corr_0[:100].max(), corr_1[:100].max(), corr_2[:100].max())+0.1)

# Retirer les labels du zoom pour éviter le fouillis
axins.set_xticks([])
axins.set_yticks([])

# Petits traits reliant la fenêtre au graphe principal
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.4")
plt.tight_layout()
plt.savefig(outdir /"toy_example_corr_long.pdf", dpi=300, bbox_inches="tight")











### 3d Plot



x = np.linspace(-3*np.pi, 3*np.pi, 400)
y = np.linspace(-2, 2, 400)
epsilon = 0.001
X, Y = np.meshgrid(x, y)
Z = F(X, Y,epsilon)



fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xticks([-10,0,10])
ax.set_yticks([-2,0,2])
ax.set_zticks([0,4])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("F(x,y)")
plt.savefig(outdir / "toy_example_fig_longer.pdf", dpi=300, bbox_inches="tight")