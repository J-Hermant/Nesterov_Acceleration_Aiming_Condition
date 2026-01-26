import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
from scipy.stats import gamma
import matplotlib.colors as mcolors


### CODE 1D

def f_quadratic_worst_case(t,mu,L):
    obj_return = np.empty(len(t))
    obj_return[np.where(t<0)] = 0.5*mu*t[np.where(t<0)[0]]**2
    obj_return[np.where(t>=0)] = 0.5*L*t[np.where(t>=0)[0]]**2
    return obj_return


def f_pl_worst_case(t,mu_0,mu,L):
    obj_return = np.empty(len(t))
    obj_return[np.where(t<0)] = 0.5*mu_0*t[np.where(t<0)[0]]**2
    obj_return[np.where(t>=0)] = 0.5*L*t[np.where(t>=0)[0]]**2 + 3*mu*np.sin(5*t[np.where(t>=0)[0]])*t[np.where(t>=0)[0]]**2
    return obj_return    

def f_1d(t,a,b):
    return 0.5*5*(t+a*np.sin(b*t))**2 
def f_1d_deriv(t,a,b):
    return 5*(t+a*np.sin(b*t))*(1+a*b*np.cos(b*t))
def f_1d_hess(t,a,b):
    return 5*(1+a*b*np.cos(b*t))*(1+a*b*np.cos(b*t)) - 5*(t+a*np.sin(b*t))*(a*b*b*np.sin(b*t))
### CODE 2D

def h(r):
    return 0.5*r**2  # décroissance radiale
def h_prime(r):  
    return r
def g(u, v,nb_f,a,b):
    # nb_f = 10
    # a,c = nprand.uniform(0,15,nb_f), nprand.uniform(0,15,nb_f)
    # b,d = nprand.uniform(-25,25,nb_f), nprand.uniform(-25,25,nb_f)
    return_var = 0
    for i in range(nb_f):
        return_var += a[i]*np.sin(b[i]*u)**2 + c[i]*np.cos(d[i]*v)**2
    return return_var/nb_f + 1 # dépend uniquement de la direction

def grad_g(u, v,nb_f,a,b):
    # nb_f = 10

    return_var = np.zeros(2)
    for i in range(nb_f):
        return_var[0] +=  2*b[i]*a[i]*np.sin(b[i]*u)*np.cos(b[i]*u) 
        return_var[1] += - 2*d[i]*c[i]*np.cos(d[i]*u)*np.sin(d[i]*u)
    return return_var/nb_f # dépend uniquement de la direction


def f(x, y,nb_f,a,b):
    r = np.sqrt(x**2 + y**2)
    # éviter la division par zéro
    u = np.where(r != 0, x/r, 0.0)
    v = np.where(r != 0, y/r, 0.0)
    return h(r) * g(u, v,nb_f,a,b)

def grad_f(x, y,nb_f,a,b):
    r = np.sqrt(x**2 + y**2)
    # éviter la division par zéro
    u = np.where(r != 0, x/r, 0.0)
    v = np.where(r != 0, y/r, 0.0)
    return np.array([u,v])/r * g(u, v,nb_f,a,b) + h(r)* grad_g(u,v,nb_f,a,b)



def g(x, y,epsilon):
    return 0.5*(0.5*x**2 - y)**2 + 0.5*epsilon *x**2

# --- Gradient ---
def grad_g(x, y,epsilon):
    return np.array([x*(0.5*x**2 - y)+epsilon*x, -(0.5*x**2 - y)])

# --- Hessian ---
def L_smooth(x, y,epsilon):
    hess_matrix = np.array([[3/2*x**2+epsilon-y, -x],[-x,1]])  
    L = np.max(np.abs(np.linalg.eig(hess_matrix)[0]))
    return L




def F(x,y,epsilon):
    return 0.5*(y-np.sin(x))**2 + 0.5*epsilon*x**2
def grad_F(x, y, epsilon):
    Fx = -(y - np.sin(x)) * np.cos(x) + epsilon * x
    Fy =  (y - np.sin(x))
    return Fx, Fy

def hess_F(x, y, eps):
    # Second derivative w.r.t x
    F_xx = np.cos(x)**2 - (y - np.sin(x))*np.sin(x) + eps
    
    # Mixed partials
    F_xy = -np.cos(x)
    F_yx = -np.cos(x)
    
    # Second derivative w.r.t y
    F_yy = 1.0
    
    H = np.array([
        [F_xx, F_xy],
        [F_yx, F_yy]
    ])
    return H

def hess_coeffs(x, y, eps):
    """
    Renvoie les coefficients a, b, c de la Hessienne 2x2 :
    H = [[a, b],
            [b, c]]
    pour F(x,y,eps) = 1/2 (y - sin x)^2 + 1/2 eps x^2
    """
    a = np.cos(x)**2 - (y - np.sin(x)) * np.sin(x) + eps
    b = -np.cos(x)
    c = 1.0
    return a, b, c

def eigen_value(x, y, eps):
    a,b,c = hess_coeffs(x, y, eps)
    trace = a + c
    delta = ((a - c) / 2)**2 + b**2
    sqrt_delta = np.sqrt(delta)

    lambda1 = trace / 2 + sqrt_delta
    lambda2 = trace / 2 - sqrt_delta

    lambda_max_abs = np.maximum(np.abs(lambda1), np.abs(lambda2))
    return lambda_max_abs


### Algorithms ###

def gd(init,n_iter,stepsize,epsilon):
    x_gd = np.empty((2,n_iter))
    f_gd = np.empty(n_iter)
    x_gd[:,0] = init
    f_gd[0] = F(x_gd[0,0],x_gd[1,0],epsilon)
    for i in range(1,n_iter):
        assign = grad_F(x_gd[0,i-1],x_gd[1,i-1],epsilon)
        grad = np.array([assign[0],assign[1]])
        x_gd[:,i] = x_gd[:,i-1] - stepsize*grad
        f_gd[i] = F(x_gd[0,i],x_gd[1,i],epsilon)
    return x_gd,f_gd

def nag(init,n_iter,stepsize,alpha,epsilon):

    x_nag = np.empty((2,n_iter))
    f_nag = np.empty(n_iter)
    corr = np.empty(n_iter)
    x_nag[:,0] = init
    x_prec = init
    f_nag[0] = F(x_nag[0,0],x_nag[1,0],epsilon)

    assign_corr = grad_F(x_nag[0,0],x_nag[1,0],epsilon)
    grad_corr = np.array([assign_corr[0],assign_corr[1]])
    norm_grad_corr = np.sqrt(np.sum(grad_corr**2))
    norm_x = np.sqrt(np.sum( x_nag[:,0]**2))
    corr[0] = np.sum(grad_corr*(x_nag[:,0]))/(norm_grad_corr*norm_x)
    for i in range(1,n_iter):
        y_nag = x_nag[:,i-1] + alpha*(x_nag[:,i-1] - x_prec)
        assign_nag = grad_F(y_nag[0],y_nag[1],epsilon)
        grad_nag = np.array([assign_nag[0],assign_nag[1]])
        x_prec =  x_nag[:,i-1]
        x_nag[:,i] = y_nag -stepsize*grad_nag
        f_nag[i] = F(x_nag[0,i],x_nag[1,i],epsilon)

        assign_corr = grad_F(x_nag[0,i],x_nag[1,i],epsilon)
        grad_corr = np.array([assign_corr[0],assign_corr[1]])
        norm_grad_corr = np.sqrt(np.sum(grad_corr**2))
        norm_x = np.sqrt(np.sum( x_nag[:,i]**2))
        corr[i] = np.sum(grad_corr*(x_nag[:,i]))/(norm_grad_corr*norm_x)
    return x_nag,f_nag,corr

