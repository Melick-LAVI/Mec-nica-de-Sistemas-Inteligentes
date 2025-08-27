import numpy as np
import matplotlib.pyplot as plt

def osc_harm(t, x, p):
    """
    Retorna a derivada do espaço de estados de um oscilador harmônico, dados:
    t: Instante no tempo                    -> float
    x: Espaço de estados no instante t      -> np.array
    p: Parâmetros no formato (ξ, ω_n, γ, Ω) -> tuple"""

    e, wn, g, O = p
    return np.array([x[1], -2*e*wn*x[1] - wn**2*x[0] + g*np.sin(O*t)])

def osc_harm_anal(t, x0, p) -> tuple:
    """
    Retorna a solução analítica do espaço de estados de um oscilador harmônico, dados:
    t:  Instante no tempo                    -> float
    x0: Valores iniciais                     -> np.array
    p:  Parâmetros no formato (ξ, ω_n, γ, Ω) -> tuple"""
    
    e, wn, g, O = p

    #Solução particular
    phi = -np.arctan2(2*e*wn*O, wn**2 - O**2)
    xp = g*np.sin(O*t + phi)/((wn**2 - O**2)**2 + (2*e*wn*O)**2)**0.5
    vp = O*g*np.cos(O*t + phi)/((wn**2 - O**2)**2 + (2*e*wn*O)**2)**0.5
    x0[0] -= g*np.sin(phi)/((wn**2 - O**2)**2 + (2*e*wn*O)**2)**0.5
    x0[1] -= O*g*np.cos(phi)/((wn**2 - O**2)**2 + (2*e*wn*O)**2)**0.5

    #Solução homogênea
    if e**2 - 1 == 0:
        r = -e*wn
        c1 = x0[0]
        c2 = x0[1] -r*x0[0]
        xh = (c1 +c2*t)*np.exp(r*t)
        vh = (r*c1 + (1 + t*r)*c2)*np.exp(r*t)

    elif e**2 -1 > 0:  
        r1 = wn*(-e +(e**2 -1)**0.5)
        r2 = -wn*(e +(e**2 -1)**0.5)
        c1 = (x0[1] -x0[0]*r2)/(r1 - r2)
        c2 = (x0[1] -x0[0]*r1)/(r2 - r1)
        xh = c1*np.exp(r1*t) + c2*np.exp(r2*t)
        vh = c1*r1*np.exp(r1*t) + c2*r2*np.exp(r2*t)

    else:
        r = -e*wn
        wd = wn*(1 -e**2)**0.5
        c1 = x0[0] 
        c2 = (x0[1] -r*x0[0])/wd
        xh = (c1*np.cos(wd*t) +c2*np.sin(wd*t))*np.exp(r*t)
        vh = ((r*c1 +wd*c2)*np.cos(wd*t) +(r*c2 -wd*c1)*np.sin(wd*t))*np.exp(r*t)

    return np.array([xp + xh, vp + vh])


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

def erro(ref, rk):
    """
    Avalia o erro da simulação numérica
    ref: Solução analítica -> np.array
    rk:  Solução numérica  -> np.array"""

    s = 0
    for i in range(len(ref)): s += np.linalg.vector_norm(ref[i] - rk[i])
    return s
    
def main() -> None:
    """
    Gera as imagens utilizadas no primeiro exercício da lista 1"""

    #Primeira simulação
    p = (0.1, 6, 10, 6)
    x0 = np.array([1, 2], dtype = "float64")
    t0 = 0
    tf = 20
    dt0 = 0.01
    
    u1 = DOPRI45(osc_harm, p, x0, t0, tf, dt0)
    u2 = RK4(osc_harm, p, x0, t0, tf, dt0)
    t = np.linspace(t0, tf, 1 + 1000*(tf-t0))
    u3 = osc_harm_anal(t, x0, p)

    plt.figure(figsize=(10, 5))
    plt.plot(u1[0], u1[1].T[0])
    plt.plot(u2[0], u2[1].T[0])
    plt.plot(t, u3[0].T)
    plt.legend(["DOPRI45", "RK4", "Analítico"])
    plt.title("Simulação do oscilador harmônico linear")
    plt.ylabel(r"$x$ (m) ")
    plt.xlabel(r"$t$ (s)")
    plt.show()

    #Segunda simulação
    p = (1, 6, 10, 6)

    u1 = DOPRI45(osc_harm, p, x0, t0, tf, dt0)
    u2 = RK4(osc_harm, p, x0, t0, tf, dt0)
    t = np.linspace(t0, tf, 1 + 1000*(tf-t0))
    u3 = osc_harm_anal(t, x0, p)

    plt.figure(figsize=(10, 5))
    plt.plot(u1[0], u1[1].T[0])
    plt.plot(u2[0], u2[1].T[0])
    plt.plot(t, u3[0].T)
    plt.legend(["DOPRI45", "RK4", "Analítico"])
    plt.title("Simulação do oscilador harmônico linear")
    plt.ylabel(r"$x$ (m) ")
    plt.xlabel(r"$t$ (s)")
    plt.show()

    #Terceira simulação
    p = (10, 6, 10, 6)

    u1 = DOPRI45(osc_harm, p, x0, t0, tf, dt0)
    u2 = RK4(osc_harm, p, x0, t0, tf, dt0)
    t = np.linspace(t0, tf, 1 + 1000*(tf-t0))
    u3 = osc_harm_anal(t, x0, p)

    plt.figure(figsize=(10, 5))
    plt.plot(u1[0], u1[1].T[0])
    plt.plot(u2[0], u2[1].T[0])
    plt.plot(t, u3[0].T)
    plt.legend(["DOPRI45", "RK4", "Analítico"])
    plt.title("Simulação do oscilador harmônico linear")
    plt.ylabel(r"$x$ (m) ")
    plt.xlabel(r"$t$ (s)")
    plt.show()

    #Análise de convergência para RK4
    p = (0.1, 6, 10, 6)
    dt = [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    err = []
    for i in dt:
        u1 = RK4(osc_harm, p, x0, t0, tf, i)
        u2 = osc_harm_anal(u1[0], x0, p)
        err.append(erro(u2.T, u1[1]))
    plt.figure(figsize=(10, 5))
    plt.loglog(dt, err, "k-o")
    plt.gca().invert_xaxis()
    plt.title("Análise de convergência para o método RK4")
    plt.xlabel("$dt$ (s)")
    plt.ylabel("$E$")
    plt.show()

    #Análise de convergencia para DOPRI45
    f = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7, 1e-7]
    err = []
    for i in f:
        u1 = DOPRI45(osc_harm, p, x0, t0, tf, i, i, i*1e-3)
        u2 = osc_harm_anal(u1[0], x0, p)
        err.append(erro(u2.T, u1[1]))
    plt.figure(figsize=(10, 5))
    plt.loglog(f, err, "k-o")
    plt.gca().invert_xaxis()
    plt.title("Análise de convergência para o método DOPRI45")
    plt.xlabel("$f$")
    plt.ylabel("$E$")
    plt.show()

    return None
