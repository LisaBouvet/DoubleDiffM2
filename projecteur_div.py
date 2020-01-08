import numpy as np

class DoubleDiff:
    def __init__(self,dt,Nstep):
        self.L = 150
        self.K = 150
        self.Lsize = self.L*2+2
        self.Ksize = self.K*2+2
        self.Nstep = Nstep
        self.dt = dt
        self.Pr = 7
        self.tau = 0.01
        self.Rp = 2
        self.u = np.zeros(((self.Ksize,self.Lsize,2)))*(1+0j) # coefs de Fourier défini comme des complexes
        self.w = np.zeros(((self.Ksize,self.Lsize,2)))*(1+0j)
        self.S = np.zeros(((self.Ksize, self.Lsize,2)))*(1 + 0j)
        self.T = np.zeros(((self.Ksize,self.Lsize,2)))*(1+0j)
        self.T[:,:,0] = (np.random.random(((self.Ksize,self.Lsize))) - 0.5)*1e-5
        self.T[:, :, 0] = np.fft.fft2(self.T[:, :, 0])
        self.T[:, :, 1] = np.fft.fft2(self.T[:, :, 1])
        self.T[:, self.L + 1,:] = 0
        self.T[self.K + 1, :,:] = 0     # perturbation initiale sur T (en mettant à 0 les fréquences de Nyquist)
        # self.E_time = np.zeros(self.Nstep)
        # self.E_kl = np.zeros(((self.Ksize,self.Lsize,self.Nstep)))
        np.savetxt('T0.csv',self.T[:,:,0])

    def matrices(self):
        l = np.zeros((self.Lsize,self.Ksize))
        l[:,0:self.K+1] = np.arange(0,self.K+1,1)
        l[:,self.K+2]=0
        l[:,self.K + 2:2 *self.K+2] = np.arange(-self.K,0,1)
        k = l.T
        Lw = - self.Pr * (k*k + l*l)
        LT = -(k*k + l*l)
        LS = -self.tau*(k*k + l*l)
        return Lw, LT, LS, l, k

    def CNAB2lin(self):
        i=0
        for n in range(self.Nstep-2):
            self.T[:,:,1] = invLT*(LT*self.T[:,:,1] - 3/2 * self.w[:,:,1] + 1/2 * self.w[:,:,0])
            self.S[:,:,1] = invLS*(LS*self.S[:,:,1] - 3/2 * self.w[:,:,1]/self.Rp + 1/2 * self.w[:,:,0]/self.Rp)
            self.w[:,:,1] = invLw*(Lw*self.w[:,:,1] + 3/2 * self.Pr*self.T[:,:,1] - 3/2 * self.Pr*self.S[:,:,1] - 1/2 * self.Pr*self.T[:,:,0] + 1/2 * self.Pr*self.S[:,:,0])
            self.u[:,:,1] = invLw * (Lw * self.u[:, :, 1])
            self.u[:,:,1] = self.u[:, :, 1] - self.u[:, :,1] * kk - self.w[:, :, 1] * kl
            self.w[:,:,1] = self.w[:, :, 1] - self.u[:, :, 1] * kl - self.w[:, :, 1] * ll
            self.w[:,:,0] = self.w[:,:,1]
            self.u[:,:,0] = self.u[:,:,1]
            self.T[:,:,0] = self.T[:, :, 1]
            self.S[:,:,0] = self.S[:, :, 1]
            print((i/(self.Nstep-2))*100)
            i+=1
        return self.w, self.T, self.S , self.u

    def CNAB2nonlin(self):
        ugradu_1, wgradw_1, ugradT_1, ugradS_1 = 0, 0, 0, 0
        i=0
        for n in range(self.Nstep - 2):
            phiw = np.fft.ifft2(self.w[:,:,1])
            phiu = np.fft.ifft2(self.u[:,:,1])
            gradux = np.fft.ifft2(1j * k * self.u[:,:,1])
            graduz = np.fft.ifft2(1j * l * self.u[:,:,1])
            gradwx = np.fft.ifft2(1j * k * self.w[:,:,1])
            gradTx = np.fft.ifft2(1j * k * self.T[:,:,1])
            gradwz = np.fft.ifft2(1j * l * self.w[:,:,1])
            gradTz = np.fft.ifft2(1j * l * self.T[:,:,1])
            gradSx = np.fft.ifft2(1j * k * self.S[:,:,1])
            gradSz = np.fft.ifft2(1j * l * self.S[:,:,1])
            ugradu = np.fft.fft2(phiu * gradux + phiw * graduz)
            wgradw = np.fft.fft2(phiw * gradwz + phiu * gradwx)
            ugradT = np.fft.fft2(phiu * gradTx + phiw * gradTz)
            ugradS = np.fft.fft2(phiu * gradSx + phiw * gradSz)
            self.w[:,:,1] = invLw*(Lw*self.w[:,:,1] + 3/2*(self.Pr*self.T[:,:,1] - self.Pr*self.S[:,:,1] - wgradw_1) - 1/2*(self.Pr*self.T[:,:,0] - self.Pr*self.S[:,:,0] - wgradw))
            self.T[:,:,1] = invLT*(LT*self.T[:,:,1] + 3/2*(-self.w[:,:,1] - ugradT_1) - 1/2*(-self.w[:,:,0] - ugradT))
            self.S[:,:,1] = invLS*(LS*self.S[:,:,1] + 3/2*(-self.w[:,:,1]/self.Rp - ugradS_1) - 1/2*(-self.w[:,:,0]/self.Rp - ugradS))
            self.u[:,:,1] = invLw*(Lw*self.u[:,:,1] + 3/2*ugradu_1 - 1/2*ugradu)
            self.u[:,:,1] = self.u[:,:,1] - self.u[:,:,1]*kk - self.w[:,:,1]*kl
            self.w[:,:,1] = self.w[:,:,1] - self.u[:,:,1]*kl - self.w[:,:,1]*ll
            self.w[:,:,0] = self.w[:,:,1]
            self.u[:,:,0] = self.u[:,:,1]
            self.T[:,:,0] = self.T[:,:,1]
            self.S[:,:,0] = self.S[:,:,1]
            ugradu_1, wgradw_1, ugradT_1, ugradS_1 = ugradu, wgradw, ugradT, ugradS
            print((i / (self.Nstep - 2)) * 100)
            i += 1
        return self.w, self.T, self.S, self.u

x = DoubleDiff(1e-6,10000)

Lw = (1 / x.dt + x.matrices()[0] / 2)
LT = (1 / x.dt + x.matrices()[1] / 2)
LS = (1 / x.dt + x.matrices()[2] / 2)
invLw = 1 / (1 / x.dt - x.matrices()[0] / 2)
invLT = 1 / (1 / x.dt - x.matrices()[1] / 2)
invLS = 1 / (1 / x.dt - x.matrices()[2] / 2)
k = x.matrices()[3]
l = x.matrices()[4]
kk = np.nan_to_num(k * k / (k * k + l * l))
kl = np.nan_to_num(k * l / (k * k + l * l))
ll = np.nan_to_num(l * l / (k * k + l * l))
#
# w,T ,S, u = x.CNAB2lin()
# np.savetxt('w.csv',w[:,:,1])
# np.savetxt('u.csv',u[:,:,1])
# np.savetxt('T.csv',T[:,:,1])
# np.savetxt('S.csv',S[:,:,1])
# -------------------------------------------

wnl,Tnl,Snl,unl= x.CNAB2nonlin()
np.savetxt('wnl.csv',wnl[:,:,1])
np.savetxt('unl.csv',unl[:,:,1])
np.savetxt('Tnl.csv',Tnl[:,:,1])
np.savetxt('Snl.csv',Snl[:,:,1])





