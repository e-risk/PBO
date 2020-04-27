import numpy as np
from scipy.integrate import simps
from StochasticMechanics import Stochastic
from Building import *
from BuildingProperties import *
from Hazards import Stationary

import matplotlib.pyplot as plt
from scipy import integrate


class PerformanceOpt(Stochastic):

    def __init__(self, power_spectrum=None, model=None, freq=None, tol=1e-5, maxiter=100, design_life=1):

        self.power_spectrum = power_spectrum
        self.model = model
        self.freq = freq
        self.tol = tol
        self.maxiter = maxiter
        self.design_life = design_life

        self.building = building
        self.columns = columns
        self.slabs = slabs
        self.core = core
        self.concrete = concrete
        self.steel = steel
        self.cost = cost

        self.ndof = building["ndof"]

        Stochastic.__init__(self, power_spectrum=power_spectrum, model=model, ndof=self.ndof, freq=freq)

    def objective_function(self, size_col=None, args=None):
        # im_max: maximum intensity measure
        # B_max: maximum barrier level

        ksi = args[0]
        im_max = args[1]
        B_max = args[2]
        gamma = args[3]
        nu = args[4]
        alpha = args[5]
        a = args[6]

        ndof = self.ndof
        self.columns = update_columns(columns=self.columns, lx=size_col, ly=size_col)
        Building = Structure(building, columns, slabs, core, concrete, steel, cost)

        if len(size_col) != ndof:
            raise ValueError('size_col is not equal to ndof!')

        initial_cost = 0
        k = []
        for i in range(ndof):
            self.columns = update_columns(columns=self.columns, lx=size_col[i], ly=size_col[i])
            Ix = size_col[i]**4/12
            Iy = Ix # Square section
            area = size_col[i]**2 #square section

            Building = Structure(building, columns, slabs, core, concrete, steel, cost)
            Cost = Costs(building, columns, slabs, core, concrete, steel, cost)
            stiffness = Building.stiffness_story()
            k.append(stiffness)

            initial_cost = initial_cost + Cost.initial_cost_stiffness(stiffness=stiffness, par0=25.55133, par1=0.33127,
                                                                      pslabs=84)
        # k[end] top floor
        k = np.array(k)
        mass = Building.mass_storey(top_story=False)
        mass_top = Building.mass_storey(top_story=True)
        m = np.ones((ndof)) * [mass]
        m[-1] = mass_top  # Top floor is m[end] - include water reservoir
        # Estimate the damping.

        c = PerformanceOpt.linear_damping(self, m=m, k=k, ksi=ksi)
        M, C, K = PerformanceOpt.create_mck(self, m=m, c=c, k=k, gamma=gamma, nu=nu, alpha=alpha, a=a)

        financial_loss_rate = PerformanceOpt.annual_financial_loss(self, M=M, C=C, K=K, stiff=k, im_max=im_max,
                                                                   B_max=B_max, size_col=size_col, gamma=gamma, nu=nu,
                                                                   alpha=alpha, a=a)

        total_loss = financial_loss_rate * self.design_life

        total_cost = initial_cost + total_loss
        print(size_col)
        print([total_cost, initial_cost, total_loss])

        return total_cost

    def annual_financial_loss(self, M=None, C=None, K=None, stiff=None, im_max=None, B_max=None, size_col=None,
                              **kwargs):

        im = im_max
        B = B_max

        CostFailure = Costs(building=building, columns=columns, slabs=slabs, core=core, concrete=concrete,
                            steel = steel, cost = cost)

        kv = wind["kv"]
        Av = wind["Av"]

        Nim = 100
        NB = 100
        imvec = np.linspace(0.00001, im_max, Nim)
        dIM = imvec[1] - imvec[0]

        integ = np.zeros(self.ndof)
        integral_IM = np.zeros((Nim, self.ndof))
        for i in range(Nim):
            im = imvec[i]
            Ps = Stationary(power_spectrum_object='windpsd', ndof=self.ndof)
            power_spectrum, ub = Ps.power_spectrum_excitation(u10=im, freq=self.freq, z=z)
            #Sto = Stochastic(power_spectrum=power_spectrum, model=self.model, ndof=ndof, freq=self.freq)

            Var, Vard = PerformanceOpt.statistical_linearization(self, M=M, C=C, K=K, tol=self.tol, maxiter=self.maxiter, **kwargs)
            Var = Var.T
            Vard = Vard.T
            Var = Var[0]
            Vard = Vard[0]

            rho = wind["rho"]
            Cd = wind["Cd"]
            L = columns["height"]
            ncolumns = columns["quantity"]
            building_area = building["height"] * building["width"]
            wind_force = rho * Cd * (building_area / 2) * (ub ** 2)
            meanY = Stochastic.linear_mean_response(stiff, wind_force)

            integral_B = []
            for j in range(self.ndof):
                B_min = max(0, meanY[j] - 1.96 * np.sqrt(Var[j]))
                B_max = max(0, meanY[j] + 1.96 * np.sqrt(Var[j]))
                Bvec = np.linspace(B_min, B_max, NB)
                dB = Bvec[1] - Bvec[0]

                up_rate = []
                for l in range(NB):
                    B = Bvec[l]
                    up_crossing_rate = ((np.sqrt(Vard[j] / Var[j])) / (2 * np.pi)) * \
                                       np.exp(-((B - meanY[j]) ** 2) / (2 * Var[j]))

                    Cf = CostFailure.cost_damage(b=B, col_size=size_col[j], L=L, ncolumns=ncolumns,
                                                 dry_wall_area=dry_wall_area)

                    rate = Cf * up_crossing_rate * Stationary.gumbel(im=im, kv=kv, Av=Av)

                    up_rate.append(rate)

                #integral_B.append(simps(np.array(up_rate), Bvec)*dIM)
                #integ[j] = integ[j] + simps(np.array(up_rate), Bvec)*dIM
                integral_IM[i,j] = simps(np.array(up_rate), Bvec)

        soma = 0
        for j in range(self.ndof):
            soma = soma + simps(integral_IM[:,j], imvec)
        # financial_loss_rate = PerformanceOpt.func_integral(B=B_max, im=im_max, M=M, C=C, K=K, stiff=stiff,
        #                                                   z=z, freq=self.freq, model=self.model, ndof=self.ndof,
        #                                                   tol=self.tol, maxiter=self.maxiter, kwargs=kwargs)

        # v = integrate.dblquad(PerformanceOpt.func_integral, 0, im_max, lambda B: 0, lambda B: B_max,
        #                  args=[M, C, K, stiff, z, self.freq, self.model, self.ndof, self.tol, self.maxiter, kwargs])

        financial_loss_rate = 60*60*24*30*12*soma
        # print(simps(simps(rate, Bvec), imvec))
        #financial_loss_rate = 0
        return financial_loss_rate

    @staticmethod
    def func_integral(B=None, im=None, M=None, C=None, K=None, stiff=None, z=None, freq=None, model=None,
                      ndof=None, tol=None, maxiter=None, kwargs=None):

        Ps = Stationary(power_spectrum_object='windpsd', ndof=ndof)
        power_spectrum, ub = Ps.power_spectrum_excitation(u10=im, freq=freq, z=z)

        Sto = Stochastic(power_spectrum=power_spectrum, model=model, ndof=ndof, freq=freq)
        Var, Vard = Sto.statistical_linearization(M=M, C=C, K=K, tol=tol, maxiter=maxiter, **kwargs)
        Var = Var.T
        Vard = Vard.T
        Var = Var[0]
        Vard = Vard[0]

        rho = wind["rho"]
        Cd = wind["Cd"]
        building_area = building["height"] * building["width"]
        wind_force = rho * Cd * (building_area / 2) * (ub ** 2)
        meanY = Stochastic.linear_mean_response(stiff, wind_force)
        up_crossing_rate = ((np.sqrt(Vard / Var)) / (2 * np.pi)) * np.exp(-((B - meanY) ** 2) / (2 * Var))
        kv = wind["kv"]
        Av = wind["Av"]

        Cf = 1000  # example only
        rate = Cf * up_crossing_rate * Stationary.gumbel(im=im, kv=kv, Av=Av)  # Annual rate

        return rate[0]

    # def initial_cost(self, m=None, c=None, k=None, num_cols_floor=None, cost_cols):
    #
    #    ndof = self.ndof
    #
    #    for i in range(ndof)
