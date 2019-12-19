import numpy as np
import matplotlib.pyplot as plt

w = np.genfromtxt('wnl.csv',dtype=complex)
T = np.genfromtxt('Tnl.csv',dtype=complex)
S = np.genfromtxt('Snl.csv',dtype=complex)
u = np.genfromtxt('unl.csv',dtype=complex)
Ekl_time = np.genfromtxt('Enl_kl_time.csv')
dt = 1e-4

Ksize = np.shape(w[0,:])
Lsize = Ksize
print(Ksize)
phiw = np.fft.ifft2(w).real
phiT = np.fft.ifft2(T).real
phiS = np.fft.ifft2(S).real
phiu = np.fft.ifft2(u).real

plt.figure(figsize = (20,7))

plt.subplot(2,4,1)#-------------------
plt.imshow(phiw,cmap='plasma')
plt.colorbar()
plt.title('w nl')
plt.subplot(2,4,2)#-------------------
plt.imshow(phiT,cmap='plasma')
plt.colorbar()
plt.title('T nl')
plt.subplot(2,4,3)#-------------------
plt.imshow(phiS,cmap='plasma')
plt.colorbar()
plt.title('S nl')
plt.subplot(2,4,4)#-------------------
plt.imshow(phiu,cmap='plasma')
plt.colorbar()
plt.title('u nl')
plt.subplot(2,4,5)#-------------------
LOGE = np.zeros(len(Ekl_time)-1)
for n in range(len(Ekl_time)-1):
    LOGE[n] = np.log(np.nan_to_num(Ekl_time[n]/Ekl_time[n+1],nan=0,posinf=0,neginf=0))/dt
plt.plot(LOGE)
plt.grid()
plt.title('$ln(E_{n+1}/E_n)/dt$')
plt.subplot(2,4,6)#-------------------
plt.plot(Ekl_time)
plt.grid()

plt.show()
