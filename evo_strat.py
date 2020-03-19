from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
from scipy.stats import rankdata
import sys
import time

def evo_strat(w_size, simulator_f, simulator_args, n_iters, n_perturbs, n_samples, n_workers=1, alpha=1, sigma=1):  
    
    executor = ProcessPoolExecutor(max_workers=n_workers)
           
    w = np.zeros(w_size)   
    perturbs = np.zeros((n_perturbs, w_size)) # save perturbations 
    
    for itr in range(n_iters): 
        tick = time.time() # start timing
        
        tasks = []        
        for i in range(n_perturbs):
            perturbs[i] = np.random.normal(0, 1, w_size) 
            for j in range(n_samples): # samples for a perturbation
                tasks.append(executor.submit(simulator_f, w + (perturbs[i] * sigma), i, simulator_args)) 
                tasks.append(executor.submit(simulator_f, w - (perturbs[i] * sigma), i + n_perturbs, simulator_args))

        reward_mean = 0
        rewards = np.zeros(n_perturbs * 2)
        for i, t in enumerate(tasks):
            out_str = str(i + 1) +  "/" + str(len(tasks))
            sys.stdout.write("\r" + out_str)
            sys.stdout.flush()
            while not t.done():
                time.sleep(0.01)
            id_num, reward = t.result()
            rewards[int(id_num)] += reward
            reward_mean += (1 /float(len(tasks))) * reward
            
        rewards /= float(n_samples)
        tock = time.time() # stop timing
        sys.stdout.write("\r")
        print(("itr:{} {} {:.4f} {:.2f} " +
               "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}").format(
                  itr, np.mean(rewards), sigma, tock-tick, 
                  np.min(rewards), np.quantile(rewards, 0.25), np.quantile(rewards, 0.5), 
                  np.quantile(rewards, 0.75), np.max(rewards))
             )
            

        # no fitness shaping
#         for i in range(n_perturbs):
#             w += alpha * (1 / float(2 * n_perturbs)) * rewards[i] * perturbs[i]
#             w += alpha * (1 / float(2 * n_perturbs)) * rewards[i + n_perturbs] * -perturbs[i]
 
        # fitness shaping 
        lmbda = float(2 * n_perturbs)
        rankings = len(rewards) + 1 - rankdata(rewards) # best value (highest reward) is #1
        utilities = np.maximum(0, np.log((lmbda / 2.) + 1) - np.log(rankings))
        utilities /= np.sum(utilities)
        utilities -= (1 /lmbda)
        
        g = np.zeros(w.shape)
        for i in range(n_perturbs):
            g += (1 / float(sigma)) * utilities[i] * perturbs[i]
            g += (1 / float(sigma)) * utilities[i + n_perturbs] * -perturbs[i]
 
        w += alpha * g
    
#         logging not used
#         np.save(savefilename, w)
            
#         log_arr = np.array([itr, np.mean(rewards), tock-tick,
#                             np.min(rewards), np.quantile(rewards, 0.25), np.quantile(rewards, 0.5),
#                             np.quantile(rewards, 0.75), np.max(rewards)])
#         old_log = np.load(savefilename + "_log.npy")
#         old_log[itr + outer_itr * n_iters] = log_arr
# #         print(itr + outer_itr * n_iters)
#         np.save(savefilename + "_log.npy", old_log)

    return w
    print("done!")
    