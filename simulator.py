import gym
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import scipy
from scipy.io import loadmat
import sys
import time

def gen_vornoi_classes(x, y, n_classes):
    """
    target is the series to fill values into.  Should be initialized
    """
    vpoints = []
    classes = np.zeros(len(x))
    for i in range(n_classes):
        vpoints.append( np.random.randint(0, len(x)) )

    for i in range(len(x)):
        dists = [np.sqrt( (xx-x[i])**2 + (yy-y[i])**2 ) for xx, yy in zip(x[vpoints], y[vpoints])]

        classes[i] = np.argmin(dists)# point it is closest to floored

    return classes, vpoints

class SpectraSimulator(gym.Env):
    def __init__(self, sp, n_classes=5, meas_cost=1):
               
        self.n_classes = n_classes
        self.meas_cost = meas_cost
        self.positions = self.setup_positions(sp)
        
        max_x_dist = np.max(self.positions[:, 0]) - np.min(self.positions[:, 0])
        max_y_dist = np.max(self.positions[:, 1]) - np.min(self.positions[:, 1])
        self.max_dist = np.sqrt(max_x_dist ** 2 + max_y_dist ** 2)
           
#         self.low = np.zeros(self.shape)
#         self.high = np.ones(self.shape) * 10000 # some constant here
#         self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=float)
                     
#         self.low2 = np.ones((self.h, self.w)) * -100
#         self.high2 = np.ones((self.h, self.w)) * 100 # some constant here
#         self.low2 = self.low2.flatten()
#         self.high2 = self.high2.flatten()
#         self.action_space = spaces.Box(low=self.low2, high=self.high2, dtype=float)
        
#         self.num_envs = 1 # what does this do??
   
    def setup_positions(self, sp):  
#         path177 = Path.cwd() / 'positions' / '177.csv'
        path177 = Path(sp) / 'positions' / '177.csv'
        locdf = pd.read_csv(path177)
        locdf = locdf.rename(columns={'Plate Y': 'Y', 'Plate X': 'X'}) 
        positions = np.vstack((locdf['X'], locdf['Y'])).T
        return positions

    def reset(self):
        classes, _ = gen_vornoi_classes(self.positions[:,0], self.positions[:,1], self.n_classes)        
        self.classes = np.array(classes)
        self.classes_meas = np.zeros(classes.size) - 1
        self.classes_pred = np.zeros(classes.size)
        
        self.n_measurements = 0
        
        self.obs = np.zeros((classes.size, 8))
        self.obs[:, :5] = 2 # make general to any shape later     
        
        self.neigh_class_pos = np.zeros((classes.size, 4, 3))
        self.neigh_class_pos[:, 0] = -1
        
        self.prev_error = np.sum(self.classes_pred != self.classes)
        
        return self.obs
        
    def step(self, action):   
        self.classes_meas[action] = self.classes[action]
        self.n_measurements += 1
        
        # observation update
        pos_diff = (self.positions - self.positions[action]) / self.max_dist
        pos_dist = np.linalg.norm(pos_diff, axis=1) 
        
        # 4 directional observations
        update_inds = np.logical_and(np.abs(pos_diff[:, 0]) > 0, pos_dist < self.obs[:, 0])
        self.neigh_class_pos[update_inds, 0, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 0, 1:] = action
        self.obs[update_inds, 0] = pos_dist[update_inds] 
        
        update_inds = np.logical_and(np.abs(-pos_diff[:, 0]) > 0, pos_dist < self.obs[:, 1])
        self.neigh_class_pos[update_inds, 1, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 1, 1:] = action
        self.obs[update_inds, 1] = pos_dist[update_inds] 
        
        update_inds = np.logical_and(np.abs(pos_diff[:, 1]) > 0, pos_dist < self.obs[:, 2])
        self.neigh_class_pos[update_inds, 2, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 2, 1:] = action
        self.obs[update_inds, 2] = pos_dist[update_inds]
        
        update_inds = np.logical_and(np.abs(-pos_diff[:, 1]) > 0, pos_dist < self.obs[:, 3])
        self.neigh_class_pos[update_inds, 3, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 3, 1:] = action
        self.obs[update_inds, 3] = pos_dist[update_inds] 
        
        # absolute closest distance
        update_inds = (pos_dist < self.obs[:, 4])
        self.classes_pred[update_inds] = self.classes[action]
        self.obs[update_inds, 4] = pos_dist[update_inds] 

        # majority count
        maj_counts = np.zeros(self.classes.size) 
        for c in range(self.n_classes):
            c_counts = np.sum(self.neigh_class_pos[:, :, 2] == c, axis=1)
            maj_counts = np.maximum(maj_counts, c_counts)
        self.obs[:, 5] = maj_counts / 4 # should be number of directions considered

        # full majority indicator
        self.obs[:, 6] = (maj_counts == 4)
        
        # meas count
        self.obs[:, 7] = self.n_measurements / self.positions.shape[0]
        
        # reward update
        error = np.sum(self.classes_pred != self.classes)
        reward = (self.prev_error - error) - self.meas_cost
        self.prev_error = error
        
        done = (self.n_measurements == 177)
        info = {}
        
        return (np.copy(self.obs), reward, done, info) 


def comparison_plot(x, y, truth, pred, measured):
    t = np.sum(measured)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, marker='s', facecolors='none', edgecolors='k', s=100)
    plt.scatter(x, y, c=pred, cmap=cm.jet, marker='s', edgecolors='k', s=100)
    plt.scatter(x[measured], y[measured], marker='s', facecolors='r', edgecolors='r', s=100)
    plt.title("Wafer Prediction with {} measurements".format(t))

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, marker='s', facecolors='none', edgecolors='k', s=100)
    plt.scatter(x, y, c=truth, cmap=cm.jet, marker='s', edgecolors='k', s=100)
    plt.title("Wafer Ground Truth")
    plt.show()
    
    
def simulator_linear(w, id_num, args, visual=False, plot_all=False): 
    # unpack args
    meas_cost = args["meas_cost"]
    
    sp = '/home/jbetterton/projects/adapt_illum_2/AdaptiveIllum_RL'
    env = SpectraSimulator(sp=sp, meas_cost=meas_cost)

    done = False
    obs = env.reset()
    measured = np.zeros(obs.shape[0], dtype=bool)
    t = 0
    r = 0
    while not done:
        ranks = obs.dot(w[:-1]) + w[-1]
        ranks[measured] = -1
        if np.max(ranks) < 0: done = True
            
        if not done:
            action = np.argmax(ranks)
            observation, reward, done, info = env.step(action)
            measured[action] = True
            t += 1 
            r += reward
        
        if visual and (plot_all or done):
            x = env.positions[:, 0]
            y = env.positions[:, 1]
            truth = env.classes
            pred = env.classes_pred
            comparison_plot(x, y, truth, pred, measured)

                        
    if visual:
        print("Measured Indices")
        print(np.where(measured))
            
    return id_num, np.sum(env.classes == env.classes_pred) - t * meas_cost

    