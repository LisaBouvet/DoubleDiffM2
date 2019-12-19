import numpy as np

wnl = np.genfromtxt('wnl.csv',dtype=complex)
Tnl = np.genfromtxt('Tnl.csv',dtype=complex)
Snl = np.genfromtxt('Snl.csv',dtype=complex)
unl = np.genfromtxt('unl.csv',dtype=complex)
Enl_kl_time = np.genfromtxt('Enl_kl_time.csv')

w = np.genfromtxt('w.csv',dtype=complex)
T = np.genfromtxt('T.csv',dtype=complex)
S = np.genfromtxt('S.csv',dtype=complex)
u = np.genfromtxt('u.csv',dtype=complex)
E_kl_time = np.genfromtxt('Ekl_time.csv')

print((1-(wnl.sum()/w.sum()))*100)
print((1-(Tnl.sum()/T.sum()))*100)
print((1-(Enl_kl_time.sum()/E_kl_time.sum()))*100)