import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nprand
from pathlib import Path


def worst_convex(x, T, t):
    x = np.asarray(x)
    assert x.shape[0] == T*t


    term1 = x[0]**2
    if t > 1:
        x_iT  = x[T-1 : (t-1)*T : T]
        x_iT1 = x[T   : (t-1)*T+1 : T]
        term1 += np.sum(((7/8)*x_iT - x_iT1)**2)

    diffs = x[1:] - x[:-1]

    mask = np.ones_like(diffs, dtype=bool)
    mask[T-1::T] = False
    term2 = np.sum(diffs[mask]**2)

    return 0.5*(term1 + term2)

def grad_worst_convex(x, t, T):

    n = T * t
    assert x.shape[0] == n
    g = np.zeros_like(x)


    i0 = np.arange(T-1, n-1, T)   
    i1 = np.arange(0, n, T)
    g[i0] += (7/8) *( (7/8)*x[i0] - x[i1][1::])
    g[i1] += - ((7/8)*np.concatenate(([0],x[i0])) - x[i1] )

    indices = np.arange(1, n + 1)
    mask_neg = (indices % T != 0)
    mask_pos = (indices % T != 1)

    idx_neg = np.nonzero(mask_neg)[0]  
    idx_pos = np.nonzero(mask_pos)[0]   

    neg = x[idx_neg]
    pos = x[idx_pos]
    d = pos - neg                   

    g[idx_pos] += d
    g[idx_neg] -= d

    return g

def g(x,t,T,y):
    return worst_convex(x,t,T) + v(x,y)

def grad_g(x,t,T,y):
    return grad_worst_convex(x,t,T) + deriv_v(x,y)

def v(x,y):
    vec_1 = (x <= (31/32)*y)*0.5*x**2 
    vec_2 =  ((31/32)*y < x)*(x <= y)*(0.5*x**2-16*(x-(31/32)*y)**2)
    vec_3 = (y < x)*(x <= (33/32)*y) * (0.5*x**2 - (1/32)*y**2 + 16*(x-(33/32)*y)**2)
    vec_4 = (x > (33/32)*y)*(0.5*x**2 -  (1/32)*y**2)
    return np.sum(vec_1 + vec_2 + vec_3 + vec_4)

def b(x,y):
    return ((31/32)*y <= x)*(x <= (33/32)*y)*(y-32*np.abs(x-y))
def deriv_v(x,y):
    deriv = x-b(x,y)
    deriv[deriv < 10**(-8)] = 0
    return deriv

def norm(x):
    return np.sqrt(np.sum(x**2))



def g_tilde(x,t,T,D,L,y):
    return (1/37)*L*g(y-x,t,T,y)

def grad_g_tilde(x,t,T,D,L,y):
    return -(1/37)*L*grad_g(y-x,t,T,y)

def run_gd(init,n_iter,stepsize,t,T,D,L,y):
    f_alg = np.empty(n_iter)
    x_alg = init.copy()
    mu_gd = np.zeros(n_iter)
    grad =  grad_g_tilde(x_alg,t,T,D,L,y)
    f_alg[0] = g_tilde(x_alg,t,T,D,L,y)
    mu_gd[0] =  0.5*norm(grad)**2/(g_tilde(x_alg,t,T,D,L,y))
    for i in range(1,n_iter):
        grad =  grad_g_tilde(x_alg,t,T,D,L,y)
        x_alg -= grad*stepsize
        mu_gd[i] = 0.5*norm(grad)**2/(g_tilde(x_alg,t,T,D,L,y))
        f_alg[i] = g_tilde(x_alg,t,T,D,L,y)
    return f_alg,mu_gd
def run_nag(init,n_iter,stepsize,alpha,t,T,D,L,y,x_ast):
    f_nag = np.empty(n_iter)
    x_nag = init.copy()
    x_nag_prec = init.copy()
    y_nag = init.copy()
    grad =  grad_g_tilde(x_nag,t,T,D,L,y)
    corr_nag = np.empty(n_iter)
    corr_nag[0] = np.sum(grad*(x_nag-x_ast))/(norm(grad)*norm(x_nag-x_ast))
    f_nag[0] =g_tilde(x_nag,t,T,D,L,y)
    mu = np.zeros(n_iter)
    mu[0] =  0.5*norm(grad)**2/(g_tilde(x_nag,t,T,D,L,y))
    for i in range(1,n_iter):
        y_nag = x_nag + alpha*(x_nag-x_nag_prec)
        x_nag_prec = x_nag
        grad_nag = grad_g_tilde(y_nag,t,T,D,L,y)
        x_nag = y_nag - grad_nag*stepsize
        mu[i] = 0.5*norm(grad_nag)**2/(g_tilde(y_nag,t,T,D,L,y))
        corr_nag[i] = np.sum(grad_nag*(y_nag-x_ast))/(norm(grad_nag)*norm(y_nag-x_ast))
        f_nag[i] = g_tilde(x_nag,t,T,D,L,y)
    return f_nag,mu,corr_nag


def run_continuized_nest(init,n_iter,gamma,gamma_p,eta,eta_p,t,T,D,L,y,x_ast):
    times=nprand.exponential(1,n_iter)

    corr_nag = np.empty(n_iter)
    mu = np.zeros(n_iter)
    x,z = init.copy(),init.copy()
    f_nag = np.empty(n_iter-1)
    grad =  grad_g_tilde(x,t,T,D,L,y)
    corr_nag[0] = np.sum(grad*(x-x_ast))/(norm(grad)*norm(x-x_ast))
    mu[0] =  0.5*norm(grad)**2/(g_tilde(x,t,T,D,L,y))
    for i in range(n_iter-1):
        time = times[i]
        param_alpha = eta/(eta+eta_p)*(1-np.exp(-(eta+eta_p)*time))
        param_beta = eta_p*(1-np.exp(-(eta+eta_p)*time))/(eta_p + eta*np.exp(-(eta+eta_p)*time))

        y_nm = (z-x)*param_alpha + x
        grad_cna = grad_g_tilde(y_nm,t,T,D,L,y)
        x = y_nm-gamma*grad_cna
        z = z + param_beta*(y_nm-z) - gamma_p *grad_cna
        mu[i+1] = 0.5*norm(grad_cna)**2/(g_tilde(y_nm,t,T,D,L,y))
        corr_nag[i+1] = np.sum(grad_cna*(y_nm-x_ast))/(norm(grad_cna)*norm(y_nm-x_ast))
        f_nag[i] = g_tilde(x,t,T,D,L,y)
    return f_nag,mu,corr_nag



mu = 1*10**-4
L = 10**3
epsilon = 10**(-10)
kappa = L/mu
C3 = 21344400/1083
C4 = 370*C3
T = int(np.floor(kappa/(37*C3)))
t = int(2*np.floor(np.log(3/(2*epsilon))/np.log(8/7)))
n = T*t
D = 20
y = (7/8)**(np.repeat(np.arange(0, t), T))
x_ast = y
init = np.zeros(n)
n_iter = 1500


### Grid Search Procedure NM
# a_p = np.linspace(0.1,2,10)
# b_p = np.linspace(0.001,0.05,10)
# c_p = np.linspace(0.001,0.5,10)
# val = 10
# for a_0 in a_p:
#     for b_0 in b_p:
#         for c_0 in c_p:
#             f_nag,blockk,blockk= run_continuized_nest(init,n_iter,9* L**(-1), np.sqrt(L)**(-1)*a_0,b_0*np.sqrt(L),np.sqrt(L)*c_0,t,T,D,L,y,x_ast)
#             val = min(val,f_nag[-1])
#             if val == f_nag[-1]:
#                 print(a_0,b_0,c_0)
#                 print(f_nag[-1])


corr_avg = 0
mu_avg = 0

corr_avg_display = np.zeros(n_iter)
mu_avg_display = np.zeros(n_iter)
n_run = 20
f_nag_avg = np.zeros(n_iter-1)
for i in range(n_run):
    f_nag,mu,corr_nag= run_continuized_nest(init,n_iter,9* L**(-1), np.sqrt(L)**(-1)*1.78,0.04*np.sqrt(L),np.sqrt(L)*0.39,t,T,D,L,y,x_ast)
    corr_avg += corr_nag.mean()
    corr_avg_display += corr_nag
    mu_avg += mu.mean()
    mu_avg_display += mu
    f_nag_avg += f_nag
f_gd,mu_gd = run_gd(init,n_iter,13* L**(-1),t,T,D,L,y)

corr_avg /= n_run
corr_avg_display /= n_run
mu_avg/=n_run
mu_avg_display/=n_run
f_nag_avg /= n_run
print("Average aiming condition : ", corr_avg)
print(" Modeled convergence rate : ",(13/3)*(mu_gd.mean()/np.sqrt(mu_avg))/np.sqrt(L)*4**(1/4))



### Code we used to estimate mu_0 and L_0. The two tested values gives a lower value approxamitvely equal to 24, a lower value approximatively equal to L_0.
### Note that if their exist tighter values, it would reinforce our results, as it would increase the lower bound on a such that we have acceleration
# vec = nprand.normal(x_ast,0.00001,n) # or nprand.normal(np.zeros(n),0.00001,n)
# print(2* g_tilde(vec,t,T,D,L,y)/norm(x_ast-vec)**2)

### PLOT
outdir = Path(__file__).parent / "figures"
outdir.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(10,5))
fontsize_legend = 16
fontsize = 14
plt.plot(np.log(f_nag_avg),lw=6,label = r"$log(f)$ using NM",color = "orange")
plt.plot(np.log(f_gd),lw=3, label =r"$log(f)$ using GD",color = "b")
plt.plot(np.arange(n_iter),np.arange(1,n_iter+1)*np.log(1-(mu_gd.mean()/L)*13)+np.log(f_gd[0]),lw=3,label="Bound (11) for GD",color = "b",linestyle ="--")
plt.plot(np.arange(n_iter),np.arange(1,n_iter+1)*np.log(1-(24/80)**(1/4)*corr_nag.mean()*np.sqrt((mu.mean()/L)*9))+np.log(f_nag[0]),lw=3,label=("Bound (12) for NM"),color = "orange",linestyle = "--")
plt.plot(np.arange(n_iter),np.arange(1,n_iter+1)*np.log(1-(10**(-4)/L)*13)+np.log(f_gd[0]),lw=3,label ="Theoretical worst case bound for GD",color = "violet")
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((-25,10),fontsize =fontsize)
# plt.ylabel("log(f)",fontsize =fontsize,labelpad=-20)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize_legend)
plt.tight_layout()
plt.savefig(outdir / "pl_worst_instance.pdf", dpi=300, bbox_inches="tight")


fig = plt.figure(figsize=(10,5))
fontsize_legend = 16
fontsize = 14

plt.plot(mu_avg_display,label=r"$\mu(\tilde x_i)$ values for NM",color="orange")
plt.plot(mu_gd,label=r"$\mu(\tilde x_i)$ values for GD",color="b",alpha = 0.5)
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.yticks((0,2),fontsize =fontsize)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize_legend)
plt.tight_layout()
plt.savefig(outdir / "mu_values.pdf", dpi=300, bbox_inches="tight")

fig = plt.figure(figsize=(10,5))
fontsize_legend = 16
fontsize = 14
plt.plot(corr_avg_display,label=r"$a(\tilde x_i)$ values for NM",color="orange")
plt.xlabel("iterations",fontsize =fontsize,labelpad=-15)
plt.xticks((0,n_iter),fontsize =fontsize)
plt.yticks((0.13,0.18),fontsize =fontsize)
plt.legend(edgecolor=(1,1,1),fontsize=fontsize_legend)
plt.tight_layout()
plt.savefig(outdir / "corr_values.pdf", dpi=300, bbox_inches="tight")