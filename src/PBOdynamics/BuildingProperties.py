import numpy as np

# number of stories
ndof = 2

# height of each story (m)
h_col = 3.65

z = np.zeros(ndof)
for i in range(ndof):
    z[i] = (i+1) * h_col

dry_wall_area = 430 # m^2 per floor

# Geometry
building = {
    "ndof": ndof,
    "height": ndof * h_col,
    "width": 18,
    "depth": 18,
}

columns = {
    "quantity": 12,
    "height": h_col,
    "lx": 0.4,
    "ly": 0.4,
    "Ix": 0.4**4/12,
    "Iy": 0.4**4/12,
    "area": 0.4*0.4,
    "v_steel": 0.01,
    "v_steel_d": 20/1000,
    "h_steel": 0.00063,
    "h_steel_d": 10/1000,
}

slabs = {
    "thickness": 0.25,
    "width": 18,
    "depth": 18,
    "steel_rate": 0.2,
}

core = {
    "quantity": 0,  # 2
    "thickness": 0.25,
    "flange": 3,
    "web": 6,
    "area": 6*0.25+(3-0.25)*0.25,
    "Ix": 15.8724, # https://calcresource.com/moment-of-inertia-channel.html
    "Iy": 2.48848,
    "distance": 2.5,
    "v_steel": 0.015,
    "v_steel_d": 15/1000,
    "h_steel": 0.04,
    "h_steel_d": 15/1000,
    "height": ndof * h_col,
}

# https://www.engineeringtoolbox.com/concrete-properties-d_1223.html
concrete = {
    "density": 2400,
    "Ec": 32000000000,
    "nu": 0.2,
    "Gc": 32000000000/(2*(1+0.2)), # Shear modulus Gc = Ec/(2(1+nu))
    "fck": 35000000,
}

steel = {
    "density": 8000, #kg/m3
    "Es": 29000000, #N/m2 = Pa
    "fy": 500000000, #Pa
}

wind = {
    "PDF": "gumbel",
    "kv": 5.83, #2.04
    "Av": 36.94, #7.04
    "rho": 1.225,
    "Cd": 1.45,
    "z0": 0.025,
    "kk": 0.4,
}

# Life-Cycle civil Engineering: proceedings of the international symposium on...
# Fabio Biodini, Dan Frangopol.
cost = {
    "IDRd": 0.003,
    "cost_IDRd": 35,
    "IDRu": 0.045,
    "cost_IDRu": 3300,

    "IDRd_eg": 0.040, # External Glazing
    "cost_IDRd_eg": 439,
    "IDRu_eg": 0.040,
    "cost_IDRu_eg": 439,

    "IDRd_dp": 0.0039, #Drywall partition
    "cost_IDRd_dp": 88,
    "IDRu_dp": 0.0085,
    "cost_IDRu_dp": 525,

    "IDRd_df": 0.0039,  # Drywall finish
    "cost_IDRd_df": 88,
    "IDRu_df": 0.0085,
    "cost_IDRu_df": 253,

    "DDI_1": 0.08,
    "DDI_2": 0.31,
    "DDI_3": 0.71,
    "DDI_4": 1.28,
    "cost_DDI_1": 8000,
    "cost_DDI_2": 22500,
    "cost_DDI_3": 34300,
    "cost_DDI_4": 34300,
}

def update_columns(columns=None, lx=None, ly=None):

    columns["lx"] = lx
    columns["ly"] = ly
    columns["area"] = lx*ly
    columns["Ix"] = lx*ly**3/12
    columns["Iy"] = ly*lx**3/12

    return columns