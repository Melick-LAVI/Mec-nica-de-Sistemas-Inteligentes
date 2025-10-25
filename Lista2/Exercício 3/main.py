import matplotlib.pyplot as plt
import numpy as np

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

def linear(t, x, p):
    """
    Retorna a derivada do espaço de estados do modelo piezoelétrico linear,
    dados:
    t: Instante no tempo                             -> float
    x: Espaço de estados [x1, x2, v] no instante t   -> np.array
       onde x1 = deslocamento, x2 = velocidade, v = tensão
    p: Parâmetros no formato (ζ, ω_n, ϑ, C_p, R, A, ω) -> tuple"""
    z, wn, th, Cp, R, A, w = p
    x1, x2, v = x

    dx1 = x2
    dx2 = -2*z*wn*x2 - wn**2*x1 + th*v + A*np.sin(w*t)
    dv  = (-v/R - th*x2) / Cp

    return np.array([dx1, dx2, dv])

def biest(t, x, p):
    """
    Retorna a derivada do espaço de estados do modelo piezoelétrico biestável,
    dados:
    t: Instante no tempo                             -> float
    x: Espaço de estados [x1, x2, v] no instante t   -> np.array
       onde x1 = deslocamento, x2 = velocidade, v = tensão
    p: Parâmetros no formato (ζ, ω_n, α, β, ϑ, C_p, R, A, ω) -> tuple"""
    z, wn, a, b, th, Cp, R, A, w = p
    x1, x2, v = x

    dx1 = x2
    dx2 = -2*z*wn*x2 + a*x1 - b*x1**3 + th*v + A*np.sin(w*t)
    dv  = (-v/R - th*x2) / Cp

    return np.array([dx1, dx2, dv])

def impact(t, x, p):
    """
    Retorna a derivada do espaço de estados do modelo piezoelétrico com batente,
    dados:
    t: Instante no tempo                             -> float
    x: Espaço de estados [x1, x2, v] no instante t   -> np.array
       onde x1 = deslocamento, x2 = velocidade, v = tensão
    p: Parâmetros no formato (ζ, ω_n, ϑ, C_p, R, A, ω, ζ_b, ω_b, g) -> tuple"""
    z, wn, th, Cp, R, A, w, zb, wb, g = p
    x1, x2, v = x

    # Dinâmica sem impacto
    dx1 = x2
    dx2 = -2*z*wn*x2 - wn**2*x1 + th*v + A*np.sin(w*t)

    # Correção se há contato com o batente
    if x1 >= g:
        dx2 = -2*(z*wn + zb*wb)*x2 - wn**2*x1 - wb**2*(x1 - g) + th*v + A*np.sin(w*t)

    dv = (-v/R - th*x2) / Cp

    return np.array([dx1, dx2, dv])

def largura_banda(freq, Pm):
    """
    Calcula a largura de banda onde Pm >= 0.5 * Pmax"""

    imax = np.argmax(Pm)
    fmax = freq[imax]
    Pmax = Pm[imax]

    filtro = Pm >= 0.5 * Pmax

    dif = np.diff(filtro.astype(int))
    starts = np.where(dif == 1)[0] + 1
    ends   = np.where(dif == -1)[0]

    if filtro[0]:
        starts = np.r_[0, starts]
    if filtro[-1]:
        ends = np.r_[ends, filtro.size - 1]

    segments = []
    faixa = 0.0
    for s, e in zip(starts, ends):
        f_start = freq[s]
        f_end   = freq[e]
        faixa += (f_end - f_start)

    return faixa, fmax, Pmax

def main():
    #Oscilador linear
    w = np.linspace(5, 45, 201)
    filt = t1 >= 20
    P = []
    for i in w:
        ap = []
        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, 2.5, i)
        t1, x1 = DOPRI45(linear, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        P.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))
    P = np.array(P)

    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador linear com $A = 2.5$ ")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.show()
    print(largura_banda(w, P))

    #Oscilador biestável
    w = np.linspace(5, 45, 101)
    P = []
    for i in w:
        ap = []
        p = (0.025, 25, 1, 10000, 0.0045, 4.2e-8, 1e5, 2.5, i)
        t1, x1 = DOPRI45(biest, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 1, 10000, 0.0045, 4.2e-8, 1e5, 5, i)
        t1, x1 = DOPRI45(biest, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 1, 10000, 0.0045, 4.2e-8, 1e5, 7.5, i)
        t1, x1 = DOPRI45(biest, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 1, 10000, 0.0045, 4.2e-8, 1e5, 9.81, i)
        t1, x1 = DOPRI45(biest, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        P.append(np.array(ap))
    P = np.array(P)
    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador biestável")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.legend(["$A = 2.5$", "$A = 5$", "$A = 7.5$", "$A = 9.81$"])
    plt.show()

    largura_banda(w, P.T[0])
    largura_banda(w, P.T[1])
    largura_banda(w, P.T[2])
    largura_banda(w, P.T[3])

    #Oscilador com impacto
    #A = 2.5
    w = np.linspace(5, 45, 101)
    P = []
    A = 2.5
    for i in w:
        ap = []
        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.001)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.0035)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.007)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.01)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        P.append(np.array(ap))
    P = np.array(P)

    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador com impacto para $A = 2.5$")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.legend(["$g = 0.001$", "$g = 0.0035$", "$g = 0.007$", "$g = 0.01$"])
    plt.show()

    largura_banda(w, P.T[0])
    largura_banda(w, P.T[1])
    largura_banda(w, P.T[2])
    largura_banda(w, P.T[3])

    #A = 5
    w = np.linspace(5, 45, 101)
    P = []
    A = 2.5
    for i in w:
        ap = []
        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.001)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.0035)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.007)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.01)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        P.append(np.array(ap))
    P = np.array(P)

    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador com impacto para $A = 5$")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.legend(["$g = 0.001$", "$g = 0.0035$", "$g = 0.007$", "$g = 0.01$"])
    plt.show()

    largura_banda(w, P.T[0])
    largura_banda(w, P.T[1])
    largura_banda(w, P.T[2])
    largura_banda(w, P.T[3])

    #A = 7.5
    w = np.linspace(5, 45, 101)
    P = []
    A = 2.5
    for i in w:
        ap = []
        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.001)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.0035)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.007)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.01)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        P.append(np.array(ap))
    P = np.array(P)

    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador com impacto para $A = 7.5$")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.legend(["$g = 0.001$", "$g = 0.0035$", "$g = 0.007$", "$g = 0.01$"])
    plt.show()

    largura_banda(w, P.T[0])
    largura_banda(w, P.T[1])
    largura_banda(w, P.T[2])
    largura_banda(w, P.T[3])

    #A = 9.81
    w = np.linspace(5, 45, 101)
    P = []
    A = 2.5
    for i in w:
        ap = []
        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.001)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.0035)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.007)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        p = (0.025, 25, 0.0045, 4.2e-8, 1e5, A, i, 0.025, 5000, 0.01)
        t1, x1 = DOPRI45(impact, p, np.array([0, 0, 0]), 0, 40, 0.001)
        filt = t1 >= 20
        t_sel = t1[filt]
        x_sel = x1[filt]
        ap.append(np.trapezoid(x_sel[:, 2]**2 / 1e5, t_sel) / (t_sel[-1] - t_sel[0]))

        P.append(np.array(ap))
    P = np.array(P)

    P = np.array(P)
    plt.figure(figsize=(10, 5))
    plt.plot(w, P)
    plt.title("Potência média para o oscilador com impacto para $A = 9.81$")
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("$P_m$ (W)")
    plt.legend(["$g = 0.001$", "$g = 0.0035$", "$g = 0.007$", "$g = 0.01$"])
    plt.show()

    largura_banda(w, P.T[0])
    largura_banda(w, P.T[1])
    largura_banda(w, P.T[2])
    largura_banda(w, P.T[3])
