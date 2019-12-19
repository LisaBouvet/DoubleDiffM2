import numpy as np
import matplotlib.pyplot as plt
j = complex(0,1)

class DoubleDiff:
    def __init__(self,dt,Nstep):
        self.L = 10
        self.K = 10
        self.Lsize = self.L*2+2
        self.Ksize = self.K*2+2
        self.Nstep = Nstep
        self.dt = dt
        self.duration = self.dt*self.Nstep  #secondes
        self.Pr = 0.1
        self.tau = 0.1
        self.Rp = 1.5
        self.u = np.ones(((self.Ksize,self.Lsize,self.Nstep)))*np.complex(0,0)  # coefs de Fourier
        self.w = np.ones(((self.Ksize,self.Lsize,self.Nstep)))*np.complex(0,0)
        self.T = np.random.random(((self.Ksize,self.Lsize,self.Nstep))) - 0.5
        self.T = np.fft.fft2(self.T)
        # self.T[:,self.L+1] = 0
        # self.T[self.K+1,:] = 0
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
        Lw = (1/self.dt + self.matrices()[0]/2)
        LT = (1/self.dt + self.matrices()[1]/2)
        LS = (1/self.dt + self.matrices()[2]/2)
        invLw = 1/(1/self.dt - self.matrices()[0]/2)
        invLT = 1/(1/self.dt - self.matrices()[1]/2)
        invLS = 1/(1/self.dt - self.matrices()[2]/2)
        k =  self.matrices()[3]
        l =  self.matrices()[4]
        kk = np.nan_to_num(k*k/(k*k + l*l))
        kl = np.nan_to_num(k*l/(k*k + l*l))
        ll = np.nan_to_num(l*l/(k*k + l*l))
        for n in range(self.Nstep-2):
            self.w[:,:,n+2] = invLw*(Lw*self.w[:,:,n+1] + 3/2 * self.Pr*self.T[:,:,n+1] - 3/2 * self.Pr*self.S[:,:,n+1] - 1/2 * self.Pr*self.T[:,:,n] + 1/2 * self.Pr*self.S[:,:,n])
            self.T[:,:,n+2] = invLT*(LT*self.T[:,:,n+1] - 3/2 * self.w[:,:,n+1] + 1/2 * self.w[:,:,n])
            self.S[:,:,n+2] = invLS*(LS*self.S[:,:,n+1] - 3/2 * self.w[:,:,n+1]/self.Rp + 1/2 * self.w[:,:,n]/self.Rp)
            self.u[:,:,n+2] = invLw*(Lw*self.u[:,:,n+1])
            self.u[:,:,n+2] = self.u[:,:,n+2] - self.u[:,:,n+2]*kk - self.w[:,:,n+2]*kl
            self.w[:,:,n+2] = self.w[:,:,n+2] - self.u[:,:,n+2]*kl - self.w[:,:,n+2]*ll
            self.E_time[n+2] = np.sqrt((np.absolute(self.w[:,:,n+2])**2).sum()+(np.absolute(self.u[:,:,n+2])**2).sum())
            self.E_kl[:,:,n+2] = np.sqrt(np.absolute(self.w[:,:,n+2])**2 + np.absolute(self.u[:,:,n+2])**2)
            #print(np.around((n/self.Nstep)*100,1),'%')
        return self.w, self.T, self.S , self.u, self.E_time, self.E_kl

    def CNAB2nonlin(self):
        Lw = (1 / self.dt + self.matrices()[0] / 2)
        LT = (1 / self.dt + self.matrices()[1] / 2)
        LS = (1 / self.dt + self.matrices()[2] / 2)
        invLw = 1 / (1 / self.dt - self.matrices()[0] / 2)
        invLT = 1 / (1 / self.dt - self.matrices()[1] / 2)
        invLS = 1 / (1 / self.dt - self.matrices()[2] / 2)
        k = self.matrices()[3]
        l = self.matrices()[4]
        kk = np.nan_to_num(k * k / (k * k + l * l))
        kl = np.nan_to_num(k * l / (k * k + l * l))
        ll = np.nan_to_num(l * l / (k * k + l * l))
        ugradu_1, wgradw_1, ugradT_1, ugradS_1 = self.nonlineaire(0)
        for n in range(self.Nstep - 2):
            ugradu, wgradw, ugradT, ugradS = self.nonlineaire(n)
            self.w[:,:,n+2] = invLw*(Lw*self.w[:,:,n+1] + 3/2*(self.Pr*self.T[:,:,n+1] - self.Pr*self.S[:,:,n+1] - wgradw_1) - 1/2*(self.Pr*self.T[:,:,n] - self.Pr*self.S[:,:,n] - wgradw))
            self.T[:,:,n+2] = invLT*(LT*self.T[:,:,n+1] + 3/2*(-self.w[:,:,n+1] - ugradT_1) - 1/2*(-self.w[:,:,n] - ugradT))
            self.S[:,:,n+2] = invLS*(LS*self.S[:,:,n+1] + 3/2*(-self.w[:,:,n+1]/self.Rp - ugradS_1) - 1/2*(-self.w[:,:,n]/self.Rp - ugradS))
            self.u[:,:,n+2] = invLw*(Lw*self.u[:,:,n+1] + 3/2*ugradu_1 - 1/2*ugradu)
            self.u[:,:,n+2] = self.u[:,:,n+2] - self.u[:,:,n+2]*kk - self.w[:,:,n+2]*kl
            self.w[:,:,n+2] = self.w[:,:,n+2] - self.u[:,:,n+2]*kl - self.w[:,:,n+2]*ll
            self.E_kl[:,:,n+2] = np.sqrt(np.absolute(self.w[:,:,n+2])**2 + np.absolute(self.u[:,:,n+2])**2)
            ugradu_1, wgradw_1, ugradT_1, ugradS_1 = ugradu, wgradw, ugradT, ugradS
            print('E=', self.E_kl[:,:,n+2].sum(), ' ', np.around((n / self.Nstep)*100, 1), '%')
            print(self.w.sum().real/self.wgradw.sum().real)
        return self.w, self.T, self.S, self.u, self.E_kl

    def nonlineaire(self,n):
        w = self.CNAB2lin()[0][:,:,n]
        T = self.CNAB2lin()[1][:,:,n]
        S = self.CNAB2lin()[2][:,:,n]
        u = self.CNAB2lin()[3][:,:,n]
        phiw = np.fft.ifft2(w).real
        phiu = np.fft.ifft2(u).real
        k =  self.matrices()[3]
        l =  self.matrices()[4]
        gradux = np.fft.ifft2(j*k*u).real
        gradwz = np.fft.ifft2(j*l*w).real
        gradTx = np.fft.ifft2(j*k*T).real
        gradTz = np.fft.ifft2(j*l*T).real
        gradSx = np.fft.ifft2(j*k*S).real
        gradSz = np.fft.ifft2(j*l*S).real
        self.ugradu = np.fft.fft2(phiu*gradux)
        self.wgradw = np.fft.fft2(phiw*gradwz)
        self.ugradT = np.fft.fft2(phiu*gradTx + phiw*gradTz)
        self.ugradS = np.fft.fft2(phiu*gradSx + phiw*gradSz)
        return self.ugradu, self.wgradw, self.ugradT, self.ugradS

x = DoubleDiff(1e-4,500)

w,T ,S, u, E_time, E_kl = x.CNAB2lin()
Ekl_time = [E_kl[:,:,n].sum() for n in range(x.Nstep)]
np.savetxt('w.csv',w[:,:,x.Nstep-1])
np.savetxt('u.csv',u[:,:,x.Nstep-1])
np.savetxt('T.csv',T[:,:,x.Nstep-1])
np.savetxt('S.csv',S[:,:,x.Nstep-1])
np.savetxt('Ekl_time.csv', Ekl_time)
np.savetxt('E_kl.csv',E_kl[:,:,x.Nstep-1])
# -------------------------------------------
wnl,Tnl,Snl,unl,Enl_kl = x.CNAB2nonlin()
np.savetxt('wnl.csv',wnl[:,:,x.Nstep-1])
np.savetxt('unl.csv',unl[:,:,x.Nstep-1])
np.savetxt('Tnl.csv',Tnl[:,:,x.Nstep-1])
np.savetxt('Snl.csv',Snl[:,:,x.Nstep-1])
np.savetxt('Enl_kl_time.csv',[Enl_kl[:,:,n].sum() for n in range(x.Nstep)])






