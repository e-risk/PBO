import numpy as np


# from BuildingProperties import *


class Structure:

    def __init__(self, building=None, columns=None, slabs=None, core=None, concrete=None, steel=None, cost=None):
        self.building = building
        self.columns = columns
        self.slabs = slabs
        self.core = core
        self.concrete = concrete
        self.steel = steel
        self.cost = cost

    def stiffness_story(self):
        area_col = self.columns["area"]
        moment_inertia_col = self.columns["Iy"]
        height_col = self.columns["height"]
        k_col = self.stiffness(area=area_col, moment_inertia=moment_inertia_col, height=height_col)

        area_core = self.core["area"]
        moment_inertia_core = self.core["Iy"]
        height_core = self.core["height"]
        k_core = self.stiffness(area=area_core, moment_inertia=moment_inertia_core, height=height_core)

        num_col = self.columns["quantity"]
        num_core = self.core["quantity"]

        k_story = num_col * k_col + num_core * k_core

        return k_story

    def stiffness(self, area=None, moment_inertia=None, height=None):
        Gc = self.concrete["Gc"]
        Ec = self.concrete["Ec"]
        Es = self.steel["Es"]
        As = self.columns["v_steel"] * area
        Ac = area - As
        E = (As * Es + Ac * Ec) / (area)

        ks = Gc * area / height
        kf = 3 * E * moment_inertia / (height ** 3)

        kt = 1 / ((1 / kf) + (1 / ks))

        return kt

    def mass_storey(self, top_story=False):

        num_col = self.columns["quantity"]
        mslab = self.mass_slab()
        mcol = self.mass_column()

        if top_story:
            mass_st = 0.5 * (num_col * mcol) + mslab
        else:
            mass_st = num_col * mcol + mslab

        return mass_st

    def mass_slab(self):

        ros = self.steel["density"]
        ro = self.concrete["density"]
        thickness = self.slabs["thickness"]
        width = self.slabs["width"]
        depth = self.slabs["depth"]

        Vs = self.slabs["steel_rate"] * thickness * width * depth
        Vc = thickness * width * depth - Vs

        mass_s = ro * Vc + ros * Vs

        return mass_s

    def mass_column(self):

        ros = self.steel["density"]
        ro = self.concrete["density"]
        height = self.columns["height"]
        area = self.columns["area"]

        As = self.columns["v_steel"] * area
        Ac = area - As

        mass_col = ro * Ac * height + ros * As * height  # +stirups

        return mass_col

    def compression(self, stiffness=None):

        # For a single column
        MPA_psi = 145.037738
        fck = self.concrete["fck"] / 1000000
        Ec = self.concrete["Ec"] / 1000000
        fy = self.steel["fy"] / 1000000
        Es = self.steel["Es"] / 1000000
        fc = fck * MPA_psi
        e_cu = 0.003
        fy = fy * MPA_psi
        Ey = Es * MPA_psi
        Ec = Ec * MPA_psi

        esy = fy / Ey
        n = Ey / Ec

        Lc = self.columns["height"]
        kc = stiffness
        Eeq = 1000000 * Ec / MPA_psi

        ros = 0.08
        ros2 = ros / 2

        b = ((Lc ** 3) * kc / (Eeq * ((1 - ros + n * ros) ** 2))) ** (1 / 4)
        b = b / 0.0254
        d_gross = b
        dc = 0.7874 * 2
        d = d_gross - dc

        As = (ros2) * b * b
        r = As / (b * d)

        Asc = As
        rc = Asc / (b * d)

        # obtain value of beta 1
        if fc <= 4000:
            b1 = 0.85

        elif fc > 4000 and fc <= 8000:
            b1 = 0.85 - 0.05 * (fc - 4000) / 1000
        else:
            b1 = 0.65

        # Initialize moment curvature matrix

        Mom = np.zeros(3)
        phi = np.zeros(3)
        # Cracking
        Ig = (b * (d_gross ** 3)) / 12
        yt = d_gross / 2
        fr = 7.5 * np.sqrt(fc)
        Mcr = fr * Ig / yt
        phi_cr = Mcr / (Ec * Ig)
        Mom[0] = Mcr

        # Yielding
        k = np.sqrt(((r + rc) * n) ** 2 + 2 * (r + rc * dc / d) * n) - (r + rc) * n
        fsc = ((k * d - dc) / (d - k * d)) * fy
        My = As * fy * d * (1 - k / 3) + Asc * fsc * ((k * d / 3) - dc)
        phi_y = esy / (d - k * d)
        Mom[1] = My

        # Ultimate
        ct = 0.5 * d
        c = 0
        cont = 0

        while abs(c / ct - 1) > 0.0002 and cont < 100:
            e_sc = ((ct - dc) / ct) * e_cu
            fsc = Ey * e_sc
            Cs = Asc * fsc
            Cc = 0.85 * fc * b * b1 * ct
            T = As * fy
            c = (T - Cs) / (0.85 * fc * b * b1)
            ct = (c + ct) / 2
            cont = cont + 1

        Mu = 0.85 * fc * b1 * c * b * (d - b1 * c / 2) + Asc * fsc * (d - dc)
        phi_u = e_cu / c
        Mom[2] = Mu

        phi[0] = phi_cr
        phi[1] = phi_y
        phi[2] = phi_u
        phi = phi / 0.0254

        return Mom, phi

    def deformation_damage_index(self, B=None, stiffness=None, Mom=None, phi=None):

        k = stiffness

        Lc = self.columns["height"]
        EI = (k * Lc ** 3) / 12
        M = 6 * EI * B / (Lc ** 2)

        My = Mom[1]  # yielding
        phiy = phi[1]
        Mu = Mom[2]  # ultimate
        phiu = phi[2]

        phim = phiu
        if M <= My:
            phim = M * phiy / My
        elif M > My and M <= Mu:
            phim = ((M - My) / (Mu - My)) * (phiu - phiy) + phiy
        elif M > Mu:
            phim = phiu

        if phim > phiy:
            ddi = (phim - phiy) / (phiu - phiy)
        elif phim < phiy:
            ddi = phim / phiy
        elif (phim > phiu):
            ddi = 1.0
        else:
            ddi = 0

        return ddi


class Costs(Structure):

    def __init__(self, building=None, columns=None, slabs=None, core=None, concrete=None, steel=None, cost=None):
        self.building = building
        self.columns = columns
        self.slabs = slabs
        self.core = core
        self.concrete = concrete
        self.steel = steel
        self.cost = cost

        Structure.__init__(self, building=building, columns=columns, slabs=slabs, core=core, concrete=concrete,
                           steel=steel, cost=cost)

    def initial_cost_stiffness(self, stiffness=None, par0=None, par1=None, pslabs=None):
        num_col = self.columns["quantity"]
        height_col = self.columns["height"]
        stiffness_kN_cm = 0.00001 * stiffness
        cost_initial = (par0 * (stiffness_kN_cm / num_col) ** par1) * num_col * height_col
        cost_initial = cost_initial + pslabs * self.slabs["width"] * self.slabs["depth"]  # price_slabs_m2*A

        return cost_initial

    def cost_damage(self, b=None, col_size=None, L=None, ncolumns=None, dry_wall_area=None):

        A_glazing = 1.5 * L
        A_bulding = 2 * L * (self.building["width"] + self.building["depth"])
        Adry = 5.95

        IDRd = self.cost["IDRd"]
        IDRu = self.cost["IDRu"]
        cIDRd = self.cost["cost_IDRd"]
        cIDRu = self.cost["cost_IDRd"]

        IDRd_eg = self.cost["IDRd_eg"]
        IDRu_eg = self.cost["IDRu_eg"]
        cIDRd_eg = self.cost["cost_IDRd_eg"]
        cIDRu_eg = self.cost["cost_IDRd_eg"]

        IDRd_dp = self.cost["IDRd_dp"]
        IDRu_dp = self.cost["IDRu_dp"]
        cIDRd_dp = self.cost["cost_IDRd_dp"]
        cIDRu_dp = self.cost["cost_IDRd_dp"]

        IDRd_df = self.cost["IDRd_df"]
        IDRu_df = self.cost["IDRu_df"]
        cIDRd_df = self.cost["cost_IDRd_df"]
        cIDRu_df = self.cost["cost_IDRd_df"]

        # COLUMNS - SLAB CONECTIONS
        bsf = IDRd * L
        bcol = IDRu * L
        csf = ncolumns * cIDRd
        ccol = ncolumns * cIDRu
        # bar(1) = datad % cost_par % bcol(i)

        # EXTERIOR GLAZING
        bsf_eg = IDRd_eg * L
        bcol_eg = IDRu_eg * L
        csf_eg = cIDRd_eg * (A_bulding / A_glazing)
        ccol_eg = cIDRu_eg * (A_bulding / A_glazing)
        # bar(2) = datad % cost_par % bcol_eg(i)

        # DRYWALL PARTITIONS
        bsf_dp = IDRd_dp * L
        bcol_dp = IDRu_dp * L
        csf_dp = cIDRd_dp * (dry_wall_area / Adry)
        ccol_dp = cIDRu_dp * (dry_wall_area / Adry)
        # bar(3) = datad % cost_par % bcol_dp(i)

        # DRYWALL FINISH
        bsf_df = IDRd_df * L
        bcol_df = IDRu_df * L
        csf_df = cIDRd_df * (dry_wall_area / Adry)
        ccol_df = cIDRu_df * (dry_wall_area / Adry)
        # bar(4) = datad % cost_par % bcol_df(i)

        if b < bsf:
            cf_cs = 0
        elif bcol > b >= bsf:
            cf_cs = ((ccol - csf) / (bcol - bsf)) * (b - bsf) + csf
        else:
            cf_cs = ccol

        if b < bsf_eg:
            cf_eg = 0
        elif bcol_eg > b >= bsf_eg:
            cf_eg = ((ccol_eg - csf_eg) / (bcol_eg - bsf_eg)) * (b - bsf_eg) + csf_eg
        else:
            cf_eg = ccol_eg

        if b < bsf_dp:
            cf_dp = 0
        elif bcol_dp > b >= bsf_dp:
            cf_dp = ((ccol_dp - csf_dp) / (bcol_dp - bsf_dp)) * (b - bsf_dp) + csf_dp
        else:
            cf_dp = ccol_dp

        if b < bsf_df:
            cf_df = 0
        elif bcol_df > b >= bsf_df:
            cf_df = ((ccol_df - csf_df) / (bcol_df - bsf_df)) * (b - bsf_df) + csf_df
        else:
            cf_df = ccol_df

        area_col = col_size**2
        moment_inertia_col = col_size**4/12

        k_col = Structure.stiffness(self, area=area_col, moment_inertia=moment_inertia_col, height=L)

        Mom, phi = Costs.compression(self, stiffness=k_col)
        ddi = Costs.deformation_damage_index(self, B=b, stiffness=k_col, Mom=Mom, phi=phi)

        DDI1 = self.cost["DDI_1"]
        DDI2 = self.cost["DDI_2"]
        DDI3 = self.cost["DDI_3"]
        DDI4 = self.cost["DDI_4"]
        cDDI1 = self.cost["cost_DDI_1"]
        cDDI2 = self.cost["cost_DDI_2"]
        cDDI3 = self.cost["cost_DDI_3"]
        cDDI4 = self.cost["cost_DDI_4"]

        if ddi < DDI1:
            cf_duc = 0
        elif ddi < DDI2 and ddi >= DDI1:
            bsf = DDI1
            bcol = DDI2
            csf = cDDI1
            ccol = cDDI2
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (b - bsf) + csf

        elif ddi < DDI3 and ddi >= DDI2:
            bsf = DDI2
            bcol = DDI3
            csf = cDDI2
            ccol = cDDI3
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (b - bsf) + csf

        elif ddi < DDI4 and ddi >= DDI3:
            bsf = DDI3
            bcol = DDI4
            csf = cDDI3
            ccol = cDDI4
            cf_duc = ((ccol - csf) / (bcol - bsf)) * (b - bsf) + csf
        else:
            cf_duc = cDDI4

        f_duc = cf_duc * ncolumns

        cf = cf_cs + cf_duc + (cf_eg + cf_dp + cf_df)

        return cf
