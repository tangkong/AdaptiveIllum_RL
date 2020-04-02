import numpy as np

from evo_strat import evo_strat
from simulator import simulator, SpectraSimulator


n_workers = 70
n_perturbs = 50
n_samples = 50
n_iters = 1000

for meas_cost in [1]:#[2, 1, 0.5, 0.2, 0.1]:
    print("\nstarting nmeas = {}\n".format(meas_cost))
    
    simulator_args = {}
    simulator_args["meas_cost"] = meas_cost
    simulator_args["nn"] = True
    simulator_args["n_classes"] = 5
    simulator_args["cell_size"] = 4.5
    w_size = 200
    w = evo_strat(w_size=w_size, simulator_f=simulator, simulator_args=simulator_args, 
                  n_iters=n_iters, n_perturbs=n_perturbs, n_samples=n_samples, n_workers=n_workers)
    
    np.save("exper1_w_bound_nn.npy".format(meas_cost), w)
    
    
    print("\nstarting nmeas = {}\n".format(meas_cost))
    
    simulator_args = {}
    simulator_args["meas_cost"] = meas_cost
    simulator_args["nn"] = False
    simulator_args["n_classes"] = 5
    simulator_args["cell_size"] = 4.5
    w_size = 17
    w = evo_strat(w_size=w_size, simulator_f=simulator, simulator_args=simulator_args, 
                  n_iters=n_iters, n_perturbs=n_perturbs, n_samples=n_samples, n_workers=n_workers)
    
    np.save("exper1_w_bound_lin.npy".format(meas_cost), w)



# v4 most recent linear 300 itrs
# v5 most recent nn     300 itrs