import numpy as np
import matplotlib.pyplot as plt
from scipy import zeros
rrho =1.5
tau =0.1
pr = 0.1

def racine (pr,tau,kh,rrho):
    A = 1
    B = (1+pr+tau)*kh**2
    C = (tau+pr+tau*pr)*kh**4+pr*(1-rrho**(-1))
    D = tau*pr*kh**6-pr*kh**2*(rrho**(-1) -tau)
    coeff = [A,B,C,D]
    return np.roots(coeff)
kh = np.linspace(0,3,(4)) #sqrt(k**2+l**2)
c=0
sig = zeros(len(kh))
omeg = zeros(len(kh))
for k in  kh:         
    rac=racine(pr,tau,k,rrho)
    indice=np.argmax(rac.real)
    sig[c]= rac[indice].real
    c=c+1

print(np.max(sig),kh[np.argmax(sig)])
plt.figure()
plt.plot(kh,sig)
plt.show()
