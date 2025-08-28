import numpy as np
import matplotlib.pyplot as plt
import sympy
import itertools

def detJf1(sigma, rho, beta):
    return [1, sigma + 1 + beta, (sigma + beta*(sigma + 1) - sigma*rho),
        sigma*beta*(1 - rho)]
    
def detJf2(sigma, rho, beta):
    return [1, (sigma+1+beta), (sigma +beta*sigma+beta), 2*beta*(rho-1)]

def anl(x: list[sympy.symbols], dxdt: list[sympy.symbols], 
        par: dict[sympy.symbols: float], solv = "num",
        chutes: list = []):
    """
    Analisa pontos de equilíbrio e avalia sua estabilidade, dados:
    x:      Espaço de estados do sistema 
                                    -> list[sympy.symbols]
    dxdt:   Derivada do espaço de estados do sistema 
                                    -> list[sympy.symbols]
    par :   Dicionário com valores numéricos para parâmetros 
                                    -> dict[sympy.symbols: float]
    solv:   Método de solução numérico ou simbólico 
                                    -> 'num' or 'sym'
    chutes: Lista de chutes iniciais para a solução numérica
                                    -> list[float]"""

    #Jacobiana
    dxdt = sympy.Matrix([i.subs(par) for i in dxdt])
    J = dxdt.jacobian(x)
    eq = []

    #Pontos de equilíbrio
    if solv == "sym": eq = sympy.solve(dxdt, x, dict=True)
    else: 
        for i in chutes:
            try:
                sol = sympy.nsolve(dxdt, x, i)
                sol = [float(s.evalf()) for s in sol]
                sol_dict = {xi: si for xi, si in zip(x, sol)}
                if not any(all(abs(sol_dict[k]-e[k]) < 1e-6 for k in x) for e in eq):
                    eq.append(sol_dict)
            except: pass
    saida = []

    #Estabilidade
    for i in eq:
        eig = np.linalg.eigvals(np.array(J.subs(i), dtype = float))
        stab = "Estável"
        if any(j.real == 0 for j in eig): stab = "Centro"
        elif any(j.real > 0 for j in eig): stab = "Instável"
  
        saida.append([i, eig, stab])
    
    return saida
    
def main():
    #Item c
    #Primeiro ponto de equilíbrio
    #Sigma
    sigma, rho, beta = 1, 1, 1
    sigma = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in sigma:
        eig = np.roots(detJf1(i, rho, beta))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(sigma), eig_real, "k")
    #plt.plot(np.log10(sigma), eig_imag, "k--")
    plt.legend(["Re(λ)"])
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel("log(σ)")
    plt.show()

    #rho
    sigma, rho, beta = 1, 1, 1
    rho = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in rho:
        eig = np.roots(detJf1(sigma, i, beta))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(rho), eig_real, "k")
    #plt.plot(np.log10(rho), eig_imag, "k--")
    plt.legend(["Re(λ)"])
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel(r"log($\rho$)")
    plt.show()

    #beta
    sigma, rho, beta = 1, 1, 1
    beta = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in beta:
        eig = np.roots(detJf1(sigma, rho, i))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(beta), eig_real, "k")
    #plt.plot(np.log10(beta), eig_imag, "k--")
    plt.legend(["Re(λ)"])
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel(r"log($\beta$)")
    plt.show()

    #Segundo e terceiro pontos de equilíbrio
    #Sigma
    sigma, rho, beta = 1, 1, 1
    sigma = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in sigma:
        eig = np.roots(detJf2(i, rho, beta))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(sigma), eig_real.T[0], "k", label = "Re(λ)")
    plt.plot(np.log10(sigma), eig_real.T[1], "k")
    plt.plot(np.log10(sigma), eig_real.T[2], "k")
    plt.plot(np.log10(sigma), eig_imag.T[0], "k--", label = "Im(λ)")
    plt.plot(np.log10(sigma), eig_imag.T[1], "k--")
    plt.plot(np.log10(sigma), eig_imag.T[2], "k--")
    plt.legend()
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel("log(σ)")
    plt.show()

    #rho
    sigma, rho, beta = 1, 1, 1
    rho = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in rho:
        eig = np.roots(detJf2(sigma, i, beta))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(rho), eig_real.T[0], "k", label = "Re(λ)")
    plt.plot(np.log10(rho), eig_real.T[1], "k")
    plt.plot(np.log10(rho), eig_real.T[2], "k")
    plt.plot(np.log10(rho), eig_imag.T[0], "k--", label = "Im(λ)")
    plt.plot(np.log10(rho), eig_imag.T[1], "k--")
    plt.plot(np.log10(rho), eig_imag.T[2], "k--")
    plt.legend()
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel(r"log($\rho$)")
    plt.show()

    #beta
    sigma, rho, beta = 1, 1, 1
    beta = np.linspace(0.01, 100, 10001)
    eig_real = []
    eig_imag = []
    for i in beta:
        eig = np.roots(detJf2(sigma, rho, i))
        real = [j.real for j in eig]
        imag = [j.imag for j in eig]
        real.sort()
        imag.sort()
        eig_real.append(real)
        eig_imag.append(imag)
    eig_real, eig_imag = np.array(eig_real), np.array(eig_imag)
    plt.figure(figsize = (10, 5))
    plt.plot(np.log10(beta), eig_real.T[0], "k", label = "Re(λ)")
    plt.plot(np.log10(beta), eig_real.T[1], "k")
    plt.plot(np.log10(beta), eig_real.T[2], "k")
    plt.plot(np.log10(beta), eig_imag.T[0], "k--", label = "Im(λ)")
    plt.plot(np.log10(beta), eig_imag.T[1], "k--")
    plt.plot(np.log10(beta), eig_imag.T[2], "k--")
    plt.legend()
    plt.title("Autovalores")
    plt.ylabel("λ")
    plt.xlabel(r"log($\beta$)")
    plt.show()

    #Item d)
    x1, x2, xp1, xp2 = sympy.symbols('x1 x2 xp1 xp2', real=True)
    zeta1, zeta2, alpha1, alpha2, beta1, beta2, rho, Omega_s = sympy.symbols(
        'zeta1 zeta2 alpha1 alpha2 beta1 beta2 rho Omega_s', real=True)

    xpp1 = -2*zeta1*xp1 + 2*zeta2*(xp2 - xp1) - (1+alpha1)*x1 - beta1*x1**3 + rho*Omega_s**2*(x2 - x1)
    xpp2 = (-2*zeta2*(xp2 - xp1) - alpha2*x2 - beta2*x2**3 - rho*Omega_s**2*(x2 - x1))/rho

    dxdt = sympy.Matrix([xp1, xp2, xpp1, xpp2])
    x = [x1, x2, xp1, xp2]
    par = {alpha1: -1, alpha2: -0.5, beta1: 1, beta2: 2, rho: 1, Omega_s: 1.5, 
           rho: 1, zeta1: 0.01, zeta2: 0.01}

    grid_x1 = [i/2 for i in range(-40, 41)]
    grid_x2 = [i/2 for i in range(-40, 41)]
    grid_xp1 = [0]
    grid_xp2 = [0]

    chutes = [tuple(c) for c in itertools.product(grid_x1, grid_x2, grid_xp1, grid_xp2)]

    print(anl(x, dxdt, par, chutes = chutes))

    return
