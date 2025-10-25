import matplotlib.pyplot as plt
import numpy as np

def main():
    (Ea, Em, theta, er, Mf, Ms, 
    As, Af, Cm, Ca, sigmas, sigmaf) = (67e3, 26.3e3, 0.55, 0.067, 9, 18.4, 34.5, 49,
                                    8, 13.8, 100, 170)
    
    #Efeito memória de forma
    #Carregamento 
    sigmal = [i for i in range(0, sigmas)]
    el = [i/Em for i in sigmal]
    T = [5 for i in sigmal]

    #Carregamento na região com transição de fase
    sigmal2 = [i for i in range(sigmas, sigmaf)]
    zeta2 = [(1 - np.cos(np.pi*(i-sigmas)/(sigmaf-sigmas)))/2 for i in sigmal2]
    el2 = []
    for i in range(len(zeta2)):
        el2.append(sigmal2[i]/Em + er*zeta2[i])
    sigmal += sigmal2
    el += el2
    T += [5 for i in zeta2]

    #Carregamento
    sigmal3 = [i for i in range(sigmaf, 200)]
    el3 = [i/Em + er for i in sigmal3]
    sigmal += sigmal3
    el += el3
    T += [5 for i in sigmal3]

    #Descarregamento
    sigmal4 = [i for i in range(200, 0, -1)]
    el4 = [i/Em + er for i in sigmal4]
    sigmal += sigmal4
    el += el4
    T += [5 for i in sigmal4]

    #Aquecimento com tensão constante 
    T += [i/10 for i in range(50, int(10*As))]
    el += [er for i in range(50, int(10*As))]
    sigmal += [0 for i in range(50, int(10*As))]
    T5 = [i/10 for i in range(int(10*As), int(10*Af))]
    zeta5 = [(1 + np.cos(np.pi*(i-As)/(Af-As)))/2 for i in T5]
    el5 = []
    sigma5 = []
    for i in range(len(T5)):
        el5.append(er*zeta5[i])
        sigma5.append(0)
    sigmal += sigma5
    el += el5
    T += T5

    #Resfriamento com tensão constante
    T += [i/10 for i in range(int(10*Af), 50, -1)]
    el += [0 for i in range(int(10*Af), 50, -1)]
    sigmal += [0 for i in range(int(10*Af), 50, -1)]

    plt.figure(figsize=(10,5))
    plt.plot(el, sigmal)
    plt.title(r"$\sigma$ (MPa) $\times$ $\epsilon$ ")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\sigma$ (MPa)")
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(el, T)
    plt.title(r"$T$ (°C) $\times$ $\epsilon$ ")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$T$ (°C)")
    plt.show()

    ax = plt.figure().add_subplot(projection='3d')
    plt.title("Efeito memória de forma")
    ax.set_xlabel(r"$T$ (°C)")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_zlabel(r"$\sigma$ (MPa)")
    ax.plot(T, el, sigmal)
    ax.set_box_aspect(None, zoom=0.85)
    ax.invert_xaxis()
    plt.show()

    #Efeito pseudoelástico
    T = 60    
    sigmas += Cm*(T - Ms)
    sigmaf += Cm*(T - Ms)

    #Carregamento 
    sigmal = [i/10 for i in range(0, int(10*sigmas))]
    el = [i/Ea for i in sigmal]

    #Carregamento na região com transição de fase
    sigmal2 = [i/10 for i in range(int(10*sigmas), int(10*sigmaf))]
    zeta2 = [(1 - np.cos(np.pi*(i-sigmas)/(sigmaf-sigmas)))/2 for i in sigmal2]
    el2 = []
    for i in range(len(zeta2)):
        el2.append(sigmal2[i]/(zeta2[i]*Em +Ea*(1-zeta2[i])) + er*zeta2[i])
    sigmal += sigmal2
    el += el2

    #Carregamento
    sigmal3 = [i/10 for i in range(int(10*sigmaf), 5500)]
    el3 = [i/Em + er for i in sigmal3]
    sigmal += sigmal3
    el += el3

    #Descarregamento
    sigmal4 = [i/10 for i in range(5500, int(10*Ca*(T-As)), -1)]
    el4 = [i/Em + er for i in sigmal4]
    sigmal += sigmal4
    el += el4

    #Descarregamento na região com transição de fase
    sigmal5 = [i/10 for i in range(int(10*Ca*(T-As)), int(10*Ca*(T-Af)), -1)]
    zeta5 = [(1 + np.cos(np.pi*(i-Ca*(T-As))/(Ca*(T-Af)-Ca*(T-As))))/2 for i in sigmal5]
    el5 = []
    for i in range(len(zeta5)):
        el5.append(sigmal5[i]/(zeta5[i]*Em +Ea*(1-zeta5[i])) + er*zeta5[i])
    sigmal += sigmal5
    el += el5

    plt.figure(figsize=(10,5))
    plt.plot(el, sigmal)
    plt.title(r"$\sigma$ (MPa) $\times$ $\epsilon$ ")
    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\sigma$ (MPa)")
    plt.show()
