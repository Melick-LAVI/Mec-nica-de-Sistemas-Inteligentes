def clas(x_poin, d, r_min = 1e-6, r_max = 1e3, ig = 4, it = 30):
    """
    Classifica a periodicidade de um mapa de Poincaré, dados:
    x_poin:  Estados do mapa de Poincaré                  -> np.array
    d:       Comprimento característico                   -> float
    r_min:   Raio mínimo de tolerância para um ponto      -> float
    r_max:   Raio máximo de tolerância para um ponto      -> float
    ig:      Máximo de iterações iguais para a heurística -> int
    it:      Máximo de iterações para a heurística        -> int"""

    x_clas = [x_poin[0]]
    r = d/100

    for i in range(1, len(x_poin)):
        cont = 1
        for j in x_clas:
            if np.linalg.norm(x_poin[i] - j) <= r: cont = 0; break

        if cont: x_clas.append(x_poin[i])

    if len(x_clas) == 1: return "Periódico", 1

    mu = 100
    cont_eq = (-1, 1)
    r = d
    while (5.5 < mu or mu < 2.5) and it:
        if mu < 3: r_min = r
        else: r_max = r

        if cont_eq[0] == len(x_clas) and cont_eq[0] != 1:
            cont_eq = (cont_eq[0], cont_eq[1] + 1)
            if cont_eq[1] == ig: return "Periódico", cont_eq[0]
        else: cont_eq = (len(x_clas), 1)

        r = (r_min + r_max)/2
        x_clas = [x_poin[0]]

        for i in range(1, len(x_poin)):
            cont = 1
            for j in x_clas:
                if np.linalg.norm(x_poin[i] - j) <= r: cont = 0; break

            if cont: x_clas.append(x_poin[i])

        it -= 1
        mu = len(x_poin)/len(x_clas)

    c = []
    for i in x_clas:
        cont = 0
        for j in x_poin:
            if np.linalg.norm(j - i) <= r: cont += 1
        c.append(cont)

    c = np.array(c)
    D = np.var(c)/np.mean(c)
    if D <= 1: return "Quase-periódico",
    else: return "Caótico",
      
def diam(x, tol = 1e-4):
    """
    Estima o diâmetro (um comprimento característico) do espaço de estados,
    dados:
    x: Vetor """
    x0 = x[0]
    x1 = 0
    ini = 1
    d = 0
    for i in range(1, len(x)):
        if ini == 1 and np.linalg.norm(x[i] - x0) > tol: ini = 0
        elif ini == 0 and np.linalg.norm(x[i] - x0) < tol: break
        else:
            if np.linalg.norm(x[i] - x0) > d:
                d = np.linalg.norm(x[i] - x1)
                x1 = x[i]

    x0 = x1
    ini = 1
    d = 0
    for i in range(1, len(x)):
        if ini == 1 and np.linalg.norm(x[i] - x0) > tol: ini = 0
        elif ini == 0 and np.linalg.norm(x[i] - x0) < tol: break
        else:
            if np.linalg.norm(x[i] - x0) > d:
                d = np.linalg.norm(x[i] - x1)

    return d

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

def osc_lin(t, x, p):
    wn, g, O = p
    return np.array([x[1], -x[0]*wn**2 +g*np.sin(O*t)])

def duffing(t, x, p):
    e, a, b, g, O = p
    return np.array([x[1], -2*e*x[1] +a*x[0] -b*x[0]**3 +g*np.sin(O*t)])

def pend(t, x, p):
    e, wn, g, O = p
    return np.array([x[1], -e*x[1] -np.sin(x[0])*wn**2 +g*np.sin(O*t)])

def poincare(t, x, T, t0):
    """
    Retorna pontos do mapa de Poincaré, dados:
    t:  Vetor de tempos da integração       -> np.array
    x:  Vetor de estados da integração      -> np.array
    T:  Período de amostragem               -> float
    t0: Tempo inicial de obtenção de pontos -> float"""

    #Listas de saída e tempos para o mapa
    t_saida, x_saida = [], []
    t_poin = t0

    for i in range(len(t)-2):
        if t[i] <= t_poin and t[i+1] > t_poin:
            #Interpolação
            idxs = [i-1, i, i+1, i+2]
            t_int = [t[i-1], t[i], t[i+1], t[i+2]]

            x_inter = []
            for j in range(x.shape[1]):
                x_near = x[idxs, j]
                L = 0.0
                for m in range(4):
                    lm = 1.0
                    for n in range(4):
                        if n != m:
                            lm *= (t_poin - t_int[n]) / (t_int[m] - t_int[n])
                    L += x_near[m] * lm
                x_inter.append(L)
            x_inter = np.array(x_inter)

            t_saida.append(t_poin)
            x_saida.append(x_inter)

            t_poin += T

    return np.array(t_saida), np.array(x_saida)
