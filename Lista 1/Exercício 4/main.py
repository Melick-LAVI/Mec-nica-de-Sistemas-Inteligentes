import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def RK4(f, p, x0, t0, tf, dt) -> tuple:
    """
    Retorna o array de valores x, do espaço de estados, utilizando o método
    de Runge-Kutta de quarta ordem (RK4), dados:
    f:  Derivada do espaço de estados  -> function
    p:  Parâmetros do sistema          -> np.array
    x0: Valores iniciais               -> np.array
    t0: Tempo de integração inicial    -> float
    tf: Tempo de integração final      -> float
    dt: Passo de integração            -> float"""

    #Atribui os valores iniciais
    t = t0
    x = x0

    #Listas de saída
    xl = [x0]
    tl = [t0]

    #Método de Runge-Kutta
    while t < tf:
        k1 = f(t,        x,  p)
        k2 = f(t + dt/2, x + k1*dt/2, p)
        k3 = f(t + dt/2, x + k2*dt/2, p)
        k4 = f(t + dt,   x + k3*dt, p)

        t += dt
        x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        tl.append(t)
        xl.append(x)

    return np.array(tl), np.array(xl)

def DOPRI45(f, p, x0, t0, tf, dt0, rtol=1e-6, atol=1e-9, dt_min=1e-8) -> tuple:
    """
    Retorna o array de valores x, do espaço de estados, utilizando o método de
    Dormand-Prince RK5(4) com passo adaptativo, dados:
    f:     Derivada do espaço de estados -> function
    p:     Parâmetros do sistema         -> np.array
    x0:    Valores iniciais              -> np.array
    t0:    Tempo de integração inicial   -> float
    tf:    Tempo de integração final     -> float
    dt0:   Passo de integração inicial   -> float
    rtol:   Tolerância de erro relativa  -> float
    atol:   Tolerância de erro absoluta  -> float
    dt_min: Passo mínimo                 -> float"""

    #Atribui os valores iniciais
    t = t0
    x = x0
    dt = dt0

    #Listas de saída
    xl = [x]
    tl = [t]

    #Método de Runge-Kutta
    while t < tf:
        if t + dt > tf: dt = tf - t
        k1 = f(t, x, p)
        k2 = f(t + dt/5,    x + k1*dt/5, p)
        k3 = f(t + 3*dt/10, x + dt*(k1*3/40 + k2*9/40), p)
        k4 = f(t + 4*dt/5,  x + dt*(k1*44/45 -k2*56/15 + k3*32/9), p)
        k5 = f(t + 8*dt/9,  x + dt*(k1*19372/6561 -k2*25360/2187 + k3*64448/6561 -k4*212/729), p)
        k6 = f(t + dt,      x + dt*(k1*9017/3168 -k2*355/33 +k3*46732/5247 + k4*49/176 -k5*5103/18656), p)
        x5 = x + dt*(k1*35/384 + k3*500/1113 + k4*125/192 - k5*2187/6784 + k6*11/84)
        k7 = f(t + dt, x5, p)
        x4 = x + dt*(k1*5179/57600 + k3*7571/16695 + k4*393/640 -k5*92097/339200 + k6*187/2100 + k7/40)

        #Estimação do erro e controle do passo
        err = (1/(atol + rtol*np.linalg.vector_norm(x5)))*np.linalg.vector_norm(x5 - x4)

        if err <= 1.0:
            t += dt
            x = x5
            tl.append(t)
            xl.append(x)
            if err == 0: dt *= 5
            else: dt *= min(5, max(0.2, 0.9*err**(-0.2)))

        else: dt *= max(0.2, 0.9 * err**(-0.2))

        if dt < dt_min: dt = dt_min

    return np.array(tl), np.array(xl)

def duffing(t, x, p):
    e, a, b = p
    return np.array([x[1], -2*e*x[1] -a*x[0] -b*x[0]**3])

def pend(t, x, p):
    wn = p
    return np.array([x[1], -np.sin(x[0])*wn**2 ])

def mult(t, x, p):
    e1, e2, a1, a2, b1, b2, rho, O = p

    x1 = x[0]
    x2 = x[1]
    v1 = x[2]
    v2 = x[3]

    ddot_x1 = (-2.0*e1*v1
               + 2.0*e2*(v2 - v1)
               - (1.0 + a1)*x1
               - b1*(x1**3)
               + rho*(O**2)*(x2 - x1))

    ddot_x2 = (-2.0*e2*(v2 - v1)
               - a2*x2
               - b2*(x2**3)
               - rho*(O**2)*(x2 - x1)) / rho

    return np.array([v1, v2, ddot_x1, ddot_x2])

def bacia_duffing(p, xlim=(-1.5, 1.5), vlim=(-1.5, 1.5),
                  n=100, tmax=50, dt0=0.01):
    e, a, b = p
    xs = np.linspace(xlim[0], xlim[1], n)
    vs = np.linspace(vlim[0], vlim[1], n)
    basin = np.zeros((n, n))

    for i, x0 in enumerate(xs):
        for j, v0 in enumerate(vs):
            _, sol = DOPRI45(duffing, p, np.array([x0, v0]), 0, tmax, dt0)
            xf = sol[-1, 0]
            if abs(xf) > 5:
                basin[j, i] = 0
            elif xf > 0.2:
                basin[j, i] = 1
            elif xf < -0.2:
                basin[j, i] = -1
            else:
                basin[j, i] = 2

    return xs, vs, basin

def bacia_pend(p, xlim=(-2*np.pi, 2*np.pi), vlim=(-4, 4), n=1000):
    wn = p
    xs = np.linspace(xlim[0], xlim[1], n)
    vs = np.linspace(vlim[0], vlim[1], n)
    basin = np.zeros((n, n))

    for i, phi0 in enumerate(xs):
        for j, dphi0 in enumerate(vs):
            E = 0.5 * dphi0**2 + wn**2 * (1 - np.cos(phi0))
            if E < 2 * wn**2:
                basin[j, i] = 0
            else:
                basin[j, i] = 1

    return xs, vs, basin

def bacia_mult(p, x1lim=(-1.5, 1.5), x2lim=(-1.5, 1.5),
               n=100, tmax=50, dt=0.01):
    x1s = np.linspace(x1lim[0], x1lim[1], n)
    x2s = np.linspace(x2lim[0], x2lim[1], n)
    basin = np.zeros((n, n))

    for i, x1_0 in enumerate(x1s):
        for j, x2_0 in enumerate(x2s):
            x0 = np.array([x1_0, x2_0, 0.0, 0.0])
            tl, sol = DOPRI45(mult, p, x0, 0.0, tmax, dt)
            xf = sol[-1, 0] 

            if abs(xf) > 5:       
                basin[j, i] = 0
            elif xf > 0.1: 
                basin[j, i] = 1
            elif xf < -0.1:      
                basin[j, i] = -1
            else:           
                basin[j, i] = 2

    return x1s, x2s, basin

def main():
    #Sistema Duffing

    p = (0.1, -1, 3)
    xs, vs, basin = bacia_duffing(p, n=3, tmax=100, dt=0.01)

    plt.title("Bacia de atração – Oscilador de Duffing")
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$\dot x_0$")

    cmap = ListedColormap(['tab:blue', 'tab:green', 'tab:orange'])
    plt.imshow(basin[::-1,:], extent=[xs[0], xs[-1], vs[0], vs[-1]],
            cmap=cmap, origin='lower', interpolation='none')

    legendas = [
        mpatches.Patch(color='tab:blue', label=r'(x,$\dot x$) = (-0,577, 0)'),
        mpatches.Patch(color='tab:orange',  label=r'(x,$\dot x$) = (+0,577, 0)')
    ]
    plt.legend(handles=legendas, loc='upper right')

    plt.tight_layout()
    plt.show()

    #Pêndulo
    wn = 1.0
    xs, vs, basin = bacia_pend(wn)

    plt.figure(figsize=(10,5))
    plt.title("Bacia de atração – Pêndulo simples")
    plt.xlabel(r"$\phi_0$")
    plt.ylabel(r"$\dot{\phi}_0$")

    cmap = ListedColormap(['tab:blue', 'tab:orange'])

    plt.imshow(basin[::-1,:], extent=[xs[0], xs[-1], vs[0], vs[-1]],
            cmap=cmap, origin='lower', interpolation='none')

    legendas = [
        mpatches.Patch(color='tab:blue', label='Movimento Oscilatório'),
        mpatches.Patch(color='tab:orange', label='Movimento de rotação indefinido')
    ]
    plt.legend(handles=legendas, loc='upper right')

    plt.tight_layout()
    plt.show()

    #Sistema multiestável
    z1 = 0.01
    z2 = 0.01
    a1 = -1.0
    a2 = -0.5
    b1 = 1.0
    b2 = 2.0
    Omega_s = 1.5
    rho = 1.0
    p = (z1, z2, a1, a2, b1, b2, rho, Omega_s)

    x1s, x2s, basin = bacia_mult(p)

    plt.figure(figsize=(10,5))
    plt.title("Bacia de atração – Sistema multiestável")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")

    cmap = ListedColormap(['tab:blue', 'tab:orange'])
    plt.imshow(basin[::-1,:], extent=[x1s[0], x1s[-1], x2s[0], x2s[-1]],
            cmap=cmap, origin='lower', interpolation='none')

    legendas = [
        mpatches.Patch(color='tab:blue', label=r'($x_1$, $x_2$) = (-0,394, -0,421)'),
        mpatches.Patch(color='tab:orange', label=r'($x_1$, $x_2$) = (+0,394, +0,421)')
    ]
    plt.legend(handles=legendas, loc='upper right')
    plt.tight_layout()
    plt.show()

