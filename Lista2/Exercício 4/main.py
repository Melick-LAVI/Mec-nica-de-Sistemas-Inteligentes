import numpy as np
import matplotlib.pyplot as plt

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

def falk(t, x, p):
    """
    Retorna a derivada do espaço de estados do oscilador baseado no modelo de
    Falk, dados:
    t : float
        Instante de tempo
    x : np.array
        Vetor de estados 
    p : tuple
        Parâmetros (ζ, a, b, T, T_m, T_a, A, ω)"""
    z, a, b, T, Tm, Ta, A, w = p
    x1, x2 = x

    dx1 = x2
    dx2 = -2*z*x2 - a*(T - Tm)*x1 + b*x1**3 - (b**2 / (4*a*(Ta - Tm))) * x1**5 + A*np.sin(w*t)

    return np.array([dx1, dx2])

def U(x, p):
    """Energia potencial do modelo de Falk"""
    z, a, b, T, Tm, Ta = p
    return a*(T - Tm)*x**2 /2 - b*x**4/ 4 + (b**2 / (4*a*(Ta - Tm))) * x**6/6

def main():
    #Energia potencial
    plt.figure(figsize = (10, 5))
    p = (0.025, 15, 6e5, 250, 287, 313)
    x1 = np.linspace(-0.1, 0.1, 201)
    u1 = U(x1, p)
    plt.plot(x1, u1)
    p = (0.025, 15, 6e5, 300, 287, 313)
    x2 = np.linspace(-0.1, 0.1, 201)
    u2 = U(x2, p)
    plt.plot(x2, u2)
    p = (0.025, 15, 6e5, 350, 287, 313)
    x3 = np.linspace(-0.1, 0.1, 201)
    u3 = U(x1, p)
    plt.plot(x1, u3)
    plt.title("Energia potencial para o modelo de Falk (1980)")
    plt.ylabel("$U$ (J)")
    plt.xlabel("$x$ (m)")
    plt.legend(["250 K", "300 K", "350 K"])
    plt.show()

    #Primeira simulação
    A = 2.5
    w = 30
    plt.figure(figsize = (10, 5))
    p = (0.025, 15, 6e5, 250, 287, 313, A, w)
    x1 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 300, 287, 313, A, w)
    x2 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 350, 287, 313, A, w)
    x3 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    plt.plot(x3[0], x3[1].T[0], "tab:green")
    plt.plot(x2[0], x2[1].T[0], "tab:orange")
    plt.plot(x1[0], x1[1].T[0],  "tab:blue")
    plt.title(r"Simulação para A = 2.5 m/s$^2$ e $\omega$ = 30 rad/s")
    plt.xlabel("$t$ (s)")
    plt.ylabel("$x$ (m)")
    plt.legend(["350 K", "300 K", "250 K"])
    plt.show()

    #Segunda simulação
    A = 5
    w = 30
    plt.figure(figsize = (10, 5))
    p = (0.025, 15, 6e5, 250, 287, 313, A, w)
    x1 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 300, 287, 313, A, w)
    x2 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 350, 287, 313, A, w)
    x3 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    plt.plot(x2[0], x2[1].T[0], "tab:orange")
    plt.plot(x3[0], x3[1].T[0], "tab:green")
    plt.plot(x1[0], x1[1].T[0],  "tab:blue")
    plt.title(r"Simulação para A = 5 m/s$^2$ e $\omega$ = 30 rad/s")
    plt.xlabel("$t$ (s)")
    plt.ylabel("$x$ (m)")
    plt.legend(["300 K", "350 K", "250 K"])
    plt.show()

    #Terceira simulação
    A = 5
    w = 50
    plt.figure(figsize = (10, 5))
    p = (0.025, 15, 6e5, 250, 287, 313, A, w)
    x1 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 300, 287, 313, A, w)
    x2 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    p = (0.025, 15, 6e5, 350, 287, 313, A, w)
    x3 = DOPRI45(falk, p, np.array([0, 0]), 0, 200, 0.001)
    plt.plot(x3[0], x3[1].T[0], "tab:green")
    plt.plot(x2[0], x2[1].T[0], "tab:orange")
    plt.plot(x1[0], x1[1].T[0],  "tab:blue")
    plt.title(r"Simulação para A = 5 m/s$^2$ e $\omega$ = 50 rad/s")
    plt.xlabel("$t$ (s)")
    plt.ylabel("$x$ (m)")
    plt.legend(["350 K", "300 K", "250 K"])
    plt.show()
