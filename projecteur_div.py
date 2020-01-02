import numpy as np
import matplotlib.pyplot as plt

class DoubleDiff:
    def __init__(self,dt,Nstep):
        self.L = 70
        self.K = 70
        self.Lsize = self.L*2+2
        self.Ksize = self.K*2+2
        self.Nstep = Nstep
        self.dt = dt
        self.duration = self.dt*self.Nstep
        self.Pr = 7
        self.tau = 0.01
        self.Rp = 2
        self.u = np.ones(((self.Ksize,self.Lsize,self.Nstep)))*np.complex(0,0)  # coefs de Fourier
        self.w = np.ones(((self.Ksize,self.Lsize,self.Nstep)))*np.complex(0,0)
        self.T = (np.random.random(((self.Ksize,self.Lsize,self.Nstep))) - 0.5)*np.complex(1,0)
        self.T[:,:,0] = np.fft.fft2(self.T[:,:,0])
        self.S = np.ones(((self.Ksize,self.Lsize,self.Nstep)))*np.complex(0,0)
        self.E_time = np.zeros(self.Nstep)
        self.E_kl = np.zeros(((self.Ksize,self.Lsize,self.Nstep))) #pour growth
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
        for n in range(self.Nstep-2):
            self.w[:,:,n+2] = invLw*(Lw*self.w[:,:,n+1] + 3/2 * self.Pr*self.T[:,:,n+1] - 3/2 * self.Pr*self.S[:,:,n+1] - 1/2 * self.Pr*self.T[:,:,n] + 1/2 * self.Pr*self.S[:,:,n])
            self.T[:,:,n+2] = invLT*(LT*self.T[:,:,n+1] - 3/2 * self.w[:,:,n+1] + 1/2 * self.w[:,:,n])
            self.S[:,:,n+2] = invLS*(LS*self.S[:,:,n+1] - 3/2 * self.w[:,:,n+1]/self.Rp + 1/2 * self.w[:,:,n]/self.Rp)
            self.u[:,:,n+2] = invLw*(Lw*self.u[:,:,n+1])
            self.u[:,:,n+2] = self.u[:,:,n+2] - self.u[:,:,n+2]*kk - self.w[:,:,n+2]*kl
            self.w[:,:,n+2] = self.w[:,:,n+2] - self.u[:,:,n+2]*kl - self.w[:,:,n+2]*ll
            self.E_time[n+2] = np.sqrt((np.absolute(self.w[:,:,n+2])**2).sum()+(np.absolute(self.u[:,:,n+2])**2).sum())
            self.E_kl[:,:,n+2] = np.sqrt(np.absolute(self.w[:,:,n+2])**2 + np.absolute(self.u[:,:,n+2])**2)
        return self.w, self.T, self.S , self.u, self.E_time, self.E_kl

    def CNAB2nonlin(self):
        ugradu_1, wgradw_1, ugradT_1, ugradS_1 = 0, 0, 0, 0
        for n in range(self.Nstep - 2):
            phiw = np.fft.ifft2(self.w[:,:,n+1])
            phiu = np.fft.ifft2(self.u[:,:,n+1])
            gradux = np.fft.ifft2(1j * k * self.u[:,:,n+1])
            graduz = np.fft.ifft2(1j * l * self.u[:,:,n+1])
            gradwx = np.fft.ifft2(1j * k * self.w[:,:,n+1])
            gradTx = np.fft.ifft2(1j * k * self.T[:,:,n+1])
            gradwz = np.fft.ifft2(1j * l * self.w[:,:,n+1])
            gradTz = np.fft.ifft2(1j * l * self.T[:,:,n+1])
            gradSx = np.fft.ifft2(1j * k * self.S[:,:,n+1])
            gradSz = np.fft.ifft2(1j * l * self.S[:,:,n+1])
            ugradu = np.fft.fft2(phiu * gradux + phiw * graduz)
            # print(np.imag(ugradu.sum()))
            # print('phy',np.imag(phiw))
            wgradw = np.fft.fft2(phiw * gradwz + phiu * gradwx)
            ugradT = np.fft.fft2(phiu * gradTx + phiw * gradTz)
            ugradS = np.fft.fft2(phiu * gradSx + phiw * gradSz)
            self.w[:,:,n+2] = invLw*(Lw*self.w[:,:,n+1] + 3/2*(self.Pr*self.T[:,:,n+1] - self.Pr*self.S[:,:,n+1] - wgradw_1) - 1/2*(self.Pr*self.T[:,:,n] - self.Pr*self.S[:,:,n] - wgradw))
            self.T[:,:,n+2] = invLT*(LT*self.T[:,:,n+1] + 3/2*(-self.w[:,:,n+1] - ugradT_1) - 1/2*(-self.w[:,:,n] - ugradT))
            self.S[:,:,n+2] = invLS*(LS*self.S[:,:,n+1] + 3/2*(-self.w[:,:,n+1]/self.Rp - ugradS_1) - 1/2*(-self.w[:,:,n]/self.Rp - ugradS))
            self.u[:,:,n+2] = invLw*(Lw*self.u[:,:,n+1] + 3/2*ugradu_1 - 1/2*ugradu)
            self.u[:,:,n+2] = self.u[:,:,n+2] - self.u[:,:,n+2]*kk - self.w[:,:,n+2]*kl
            self.w[:,:,n+2] = self.w[:,:,n+2] - self.u[:,:,n+2]*kl - self.w[:,:,n+2]*ll
            self.E_kl[:,:,n+2] = np.sqrt(np.absolute(self.w[:,:,n+2])**2 + np.absolute(self.u[:,:,n+2])**2)
            ugradu_1, wgradw_1, ugradT_1, ugradS_1 = ugradu, wgradw, ugradT, ugradS
            print('E=', self.E_kl[:,:,n+2].sum(), ' ', np.around((n / self.Nstep)*100, 2), '%')
        return self.w, self.T, self.S, self.u, self.E_kl

x = DoubleDiff(1e-6,5000)

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


# w,T ,S, u, E_time, E_kl = x.CNAB2lin()
# Ekl_time = [E_kl[:,:,n].sum() for n in range(x.Nstep)]
# np.savetxt('w.csv',w[:,:,x.Nstep-1])
# np.savetxt('u.csv',u[:,:,x.Nstep-1])
# np.savetxt('T.csv',T[:,:,x.Nstep-1])
# np.savetxt('S.csv',S[:,:,x.Nstep-1])
# np.savetxt('Ekl_time.csv', Ekl_time)
# np.savetxt('E_kl.csv',E_kl[:,:,x.Nstep-1])
# -------------------------------------------


wnl,Tnl,Snl,unl,Enl_kl = x.CNAB2nonlin()
np.savetxt('wnl.csv',wnl[:,:,x.Nstep-1])
np.savetxt('unl.csv',unl[:,:,x.Nstep-1])
np.savetxt('Tnl.csv',Tnl[:,:,x.Nstep-1])
np.savetxt('Snl.csv',Snl[:,:,x.Nstep-1])
np.savetxt('Enl_kl_time.csv',[Enl_kl[:,:,n].sum() for n in range(x.Nstep)])






