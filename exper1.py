import numpy as np

from evo_strat import evo_strat
from simulator import simulator_linear, SpectraSimulator


n_workers = 2 #70
n_perturbs = 50
n_samples = 50
n_iters = 100


for meas_cost in [2, 1, 0.5, 0.2, 0.1]:
    print("\nstarting nmeas = {}\n".format(meas_cost))
    
    simulator_args = {}
    simulator_args["meas_cost"] = meas_cost
    w = evo_strat(w_size=9, simulator_f=simulator_linear, simulator_args=simulator_args, 
                  n_iters=n_iters, n_perturbs=n_perturbs, n_samples=n_samples, n_workers=n_workers)
    
    np.save("exper1_w_nmeas_{}_v2.npy".format(meas_cost), w)



