import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys

Ua = [5,10,15,20,25,30]
ia = [0.555,0.685,0.760, 0.810, 0.870, 0.900]
omega = [0,29.5, 65, 100, 139, 176]
Uf = 10
Rf = 320
Ra = Ua[0]/ia[0]
print("Ra =", Ra)

kphi = 0
for i in range(len(Ua)):
    kphi += (Ua[i] - Ra*ia[i]) * 1/omega[i]
kphi /= len(Ua)
print("kphi =", kphi)

dMaf_dteta = kphi * Rf/Uf
print("dMaf/dteta =", dMaf_dteta)

# Résolution d'un système de 3 équations à 3 inconnues
# Exemple : a1*x + b1*y + c1*z = d1, etc.
A = np.array([[1, -1, 3],
              [1,  0, 1],
              [1,  2, -2]])
B = np.array([5, 2, 7])

solution = np.linalg.solve(A, B)
print("Solution du système : x = {:.3f}, y = {:.3f}, z = {:.3f}".format(*solution))

# Affichage des résultats
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(omega, Ua, 'o-', label='Tension (Ua)')
plt.plot(omega, [Ra*i for i in ia], 's--', label='Chute de tension (Ra*ia)')
plt.xlabel('Vitesse angulaire (omega)')
plt.ylabel('Tension (V)')
plt.title('Caractéristique de la machine')
plt.legend()
plt.grid()