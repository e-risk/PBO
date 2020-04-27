
import matplotlib.pyplot as plt
import numpy as np
from StochasticMechanics import Stochastic
from scipy.optimize import minimize
from Optimization import PerformanceOpt
from Hazards import Stationary
from Building import *
from BuildingProperties import *

freq = np.linspace(0.00001, 20, 500)

gamma = np.ones((ndof)) * [0.5]
nu = np.ones((ndof)) * [0.5]
alpha = np.ones((ndof)) * [1]

m = np.ones((ndof)) * [1]
c = np.ones((ndof)) * [1]
k = np.ones((ndof)) * [200]
a = np.ones((ndof)) * [0.01]
ksi = [0.05, 0.05]

im_max = 30
B_max = 1

#S1 = np.ones(ndof)
#Ps = Stationary(power_spectrum_object='white_noise', ndof=ndof)
#power_spectrum = Ps.power_spectrum_excitation(freq=freq, S0=S1)

# Von Karman
Ps = Stationary(power_spectrum_object='windpsd', ndof=ndof)
power_spectrum, U = Ps.power_spectrum_excitation(u10=6.2371, freq=freq, z=z)

#plt.semilogy(freq/(2*np.pi), power_spectrum[:,0])
#plt.show()

#columns["area"] = 0.001
#columns.update({"area": 0.001})

ks=[]
ms=[]
msf=[]
cost=[]
nlc=100
lc = np.linspace(0.05,2,nlc)



#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#fig.suptitle('Mass and Stiffness')
#ax1.plot(lc,ms)
#ax1.plot(lc,msf)
#ax2.plot(lc,ks)
#ax3.plot(ks,cost)
#plt.show()

columns = update_columns(columns=columns, lx=0.4, ly=0.4)

Building = Structure(building, columns, slabs, core, concrete, steel)
k_story = Building.stiffness_story()
m_story = Building.mass_storey(top_story=False)
m_story_f = Building.mass_storey(top_story=True)

k = np.ones(ndof) * [k_story]
m = np.ones(ndof) * [m_story]
m[-1] = m_story_f

length = 0.3
size_col = np.ones(ndof) * [length]

Opt = PerformanceOpt(power_spectrum=power_spectrum, model='bouc_wen', freq=freq, tol=1e-5, maxiter=100,
                     design_life=50)

#total_cost = Opt.objective_function(size_col=size_col, ksi=ksi, im_max=im_max, B_max=B_max, gamma=gamma, nu=nu,
#                                    alpha=alpha, a=a)

#print(total_cost)
#bnds = []
#for i in range(ndof):
#    bnds.append((0.1, 0.5))
#
#bnds=tuple(bnds)
#res = minimize(Opt.objective_function, x0=size_col, args=[ksi, im_max, B_max, gamma, nu, alpha, a], bounds=bnds)

b0 = np.linspace(0.1,0.5,10)
b1 = np.linspace(0.1,0.5,10)
B0, B1 = np.meshgrid(b0, b1)
args=[ksi, im_max, B_max, gamma, nu, alpha, a]
tc = np.zeros((10,10))
for i in range(len(b0)):
    print(i)
    for j in range(len(b1)):
        size_col = np.array([b0[i], b1[j]])
        tc[i,j] = Opt.objective_function(size_col=size_col, args=args)


Z = tc.reshape(B0.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B0, B1, Z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
