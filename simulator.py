from collections import OrderedDict
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
import torch

class SpectraSimulator(gym.Env):
    def __init__(self, sp, n_classes=5, meas_cost=1, boundary_obj=True, test_data=None, cell_size=4.5):
        
        self.cell_size = cell_size 
        self.boundary_obj = boundary_obj
               
        self.n_classes = n_classes
        self.meas_cost = meas_cost
       
        self.test_data_flag = (test_data is not None)
        if test_data is None:
            self.positions = self.setup_positions(sp)
        else:
            self.positions = np.vstack((test_data['X'], test_data['Y'])).T
            self.classes = test_data['class']
            
        self.min_x = np.min(self.positions[:, 0])
        self.min_y = np.min(self.positions[:, 1])
        self.max_x_dist = np.max(self.positions[:, 0]) - self.min_x
        self.max_y_dist = np.max(self.positions[:, 1]) - self.min_y
        self.max_dist = np.sqrt(self.max_x_dist ** 2 + self.max_y_dist ** 2)   
           
#         self.low = np.zeros(self.shape)
#         self.high = np.ones(self.shape) * 10000 # some constant here
#         self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=float)
                     
#         self.low2 = np.ones((self.h, self.w)) * -100
#         self.high2 = np.ones((self.h, self.w)) * 100 # some constant here
#         self.low2 = self.low2.flatten()
#         self.high2 = self.high2.flatten()
#         self.action_space = spaces.Box(low=self.low2, high=self.high2, dtype=float)
        
#         self.num_envs = 1 # what does this do?
   
    def setup_positions(self, sp):  
        path177 = Path(sp) / 'positions' / '177.csv'
        locdf = pd.read_csv(path177)
        locdf = locdf.rename(columns={'Plate Y': 'Y', 'Plate X': 'X'}) 
        positions = np.vstack((locdf['X'], locdf['Y'])).T
        return positions
    
    def create_grid(self):
        self.grid = -1 * np.ones((int(self.max_x_dist / self.cell_size) + 1, 
                                  int(self.max_y_dist / self.cell_size) + 1)) 
        for i, c in enumerate(self.classes):
            x, y = self.positions[i] 
            self.grid[int((x - self.min_x) / self.cell_size), 
                      int((y - self.min_y) / self.cell_size)] = c
    
    def find_boundary_points(self, wafer_edge=True): 
        self.bound_grid = np.zeros(self.grid.shape, dtype=bool)
        self.wafer_edge_grid = np.zeros(self.grid.shape, dtype=bool)
        
        # booundaries 
        diff_x = self.grid[1:, :] - self.grid[:-1, :]
        wafer_inter_x = np.logical_and(self.grid[1:, :] != -1, self.grid[:-1, :] != -1)
        bound_x = np.logical_and(wafer_inter_x, diff_x != 0)    
        self.bound_grid[1:, :] = np.logical_or(self.bound_grid[1:, :], bound_x)
        self.bound_grid[:-1, :] = np.logical_or(self.bound_grid[:-1, :], bound_x)
        
        diff_y = self.grid[:, 1:] - self.grid[:, :-1]
        wafer_inter_y = np.logical_and(self.grid[:, 1:] != -1, self.grid[:, :-1] != -1)
        bound_y = np.logical_and(wafer_inter_y, diff_y != 0)    
        self.bound_grid[:, 1:] = np.logical_or(self.bound_grid[:, 1:], bound_y)
        self.bound_grid[:, :-1] = np.logical_or(self.bound_grid[:, :-1], bound_y)
                
        self.boundaries = np.zeros(self.classes.shape, dtype=bool)        
        for i in range(self.positions.shape[0]):
            x, y = self.positions[i]
            self.boundaries[i] = self.bound_grid[int((x - self.min_x) / self.cell_size), 
                                                 int((y - self.min_y) / self.cell_size)]  
            
        # wafer edge
        self.wafer_edge_grid[:-1, :] = np.logical_or(self.wafer_edge_grid[:-1, :], (self.grid[1:, :] == -1))
        self.wafer_edge_grid[1:, :] = np.logical_or(self.wafer_edge_grid[1:, :], (self.grid[:-1, :] == -1))
        self.wafer_edge_grid[:, :-1] = np.logical_or(self.wafer_edge_grid[:, :-1], (self.grid[:, 1:] == -1))
        self.wafer_edge_grid[:, 1:] = np.logical_or(self.wafer_edge_grid[:, 1:], (self.grid[:, :-1] == -1))
        self.wafer_edge_grid[:, 0] = 1
        self.wafer_edge_grid[:, -1] = 1
        self.wafer_edge_grid[0, :] = 1
        self.wafer_edge_grid[-1, :] = 1
        
        self.wafer_edge = np.zeros(self.classes.shape, dtype=bool) 
        for i in range(self.positions.shape[0]):
            x, y = self.positions[i]
            self.wafer_edge[i] = self.wafer_edge_grid[int((x - self.min_x) / self.cell_size), 
                                                      int((y - self.min_y) / self.cell_size)]  
         

    def reset(self):    
        if not self.test_data_flag: 
            classes, _ = gen_vornoi_classes(self.positions[:,0], self.positions[:,1], self.n_classes)    
            self.classes = np.array(classes)
        self.classes_meas = np.zeros(self.classes.size) - 1
        self.classes_pred = np.zeros(self.classes.size)
        
        self.n_measurements = 0
        
        self.create_grid()
        self.find_boundary_points()
        
        self.obs = np.zeros((self.classes.size, 16))
        self.obs[:, :5] = 2 # make general to any shape later     
        self.obs[:, 8:12] = 2
        
        # 4 directional, 4 lines
        self.neigh_class_pos = np.zeros((self.classes.size, 8, 3))
        self.neigh_class_pos[:, 0] = -1
        
        self.prev_error = np.sum(self.classes_pred != self.classes)
        
        self.ordered_measurements = []
        
        return self.obs
        
    def step(self, action):   
        self.classes_meas[action] = self.classes[action]
        self.n_measurements += 1
        self.ordered_measurements.append(self.positions[action])
        
        # observation update
        pos_diff = (self.positions - self.positions[action]) / self.max_dist
        pos_dist = np.linalg.norm(pos_diff, axis=1) 
        
        # 4 directional observations
        update_inds = np.logical_and(pos_diff[:, 0] > 0, pos_dist < self.obs[:, 0])
        self.neigh_class_pos[update_inds, 0, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 0, 1:] = action
        self.obs[update_inds, 0] = pos_dist[update_inds] 
        
        update_inds = np.logical_and(pos_diff[:, 0] < 0, pos_dist < self.obs[:, 1])
        self.neigh_class_pos[update_inds, 1, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 1, 1:] = action
        self.obs[update_inds, 1] = pos_dist[update_inds] 
        
        update_inds = np.logical_and(pos_diff[:, 1] > 0, pos_dist < self.obs[:, 2])
        self.neigh_class_pos[update_inds, 2, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 2, 1:] = action
        self.obs[update_inds, 2] = pos_dist[update_inds]
        
        update_inds = np.logical_and(pos_diff[:, 1] < 0, pos_dist < self.obs[:, 3])
        self.neigh_class_pos[update_inds, 3, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, 3, 1:] = action
        self.obs[update_inds, 3] = pos_dist[update_inds] 
        
        # absolute closest distance
        update_inds = (pos_dist < self.obs[:, 4])
        self.classes_pred[update_inds] = self.classes[action]
        self.obs[update_inds, 4] = pos_dist[update_inds] 

        # majority count 
        # TODO CHECK DEFAULT
        maj_counts = np.zeros(self.classes.size) 
        for c in range(self.n_classes):
            c_counts = np.sum(self.neigh_class_pos[:, :, 2] == c, axis=1)
            maj_counts = np.maximum(maj_counts, c_counts)
        self.obs[:, 5] = maj_counts / 4 # should be number of directions considered

        # full majority indicator
        self.obs[:, 6] = (maj_counts == 4)
        
        # meas count
        self.obs[:, 7] = self.n_measurements / self.positions.shape[0]
        
        # new features to help with boundary - line features 
        neigh_ind = 4
        obs_ind = 8
        eligible_inds = np.logical_and(pos_diff[:, 0] > 0, pos_diff[:, 1] == 0)
        update_inds = np.logical_and(eligible_inds, pos_dist < self.obs[:, obs_ind])
        self.neigh_class_pos[update_inds, neigh_ind, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, neigh_ind, 1:] = action
        self.obs[update_inds, obs_ind] = pos_dist[update_inds] 
           
        neigh_ind += 1
        obs_ind += 1
        eligible_inds = np.logical_and(pos_diff[:, 0] < 0, pos_diff[:, 1] == 0)
        update_inds = np.logical_and(eligible_inds, pos_dist < self.obs[:, obs_ind])
        self.neigh_class_pos[update_inds, neigh_ind, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, neigh_ind, 1:] = action
        self.obs[update_inds, obs_ind] = pos_dist[update_inds] 
        
        obs_ind += 1
        eligible_inds = np.logical_and(pos_diff[:, 0] == 0, pos_diff[:, 1] > 0)
        update_inds = np.logical_and(eligible_inds, pos_dist < self.obs[:, obs_ind])
        self.neigh_class_pos[update_inds, neigh_ind, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, neigh_ind, 1:] = action
        self.obs[update_inds, obs_ind] = pos_dist[update_inds] 
                                     
        obs_ind += 1
        eligible_inds = np.logical_and(pos_diff[:, 0] == 0, pos_diff[:, 1] < 0)
        update_inds = np.logical_and(eligible_inds, pos_dist < self.obs[:, obs_ind])
        self.neigh_class_pos[update_inds, neigh_ind, 0] = self.classes[action]
        self.neigh_class_pos[update_inds, neigh_ind, 1:] = action
        self.obs[update_inds, obs_ind] = pos_dist[update_inds] 
        
        # new features to help with boundary - line features agreement
        #(classes are the same and they're not both boundaries then theres a connected line
        x_class_diff = self.neigh_class_pos[:, 4, 0] - self.neigh_class_pos[:, 5, 0]
        self.obs[:, 12] = np.logical_and(x_class_diff == 0, self.neigh_class_pos[:, 4, 0] != -1)
        
        y_class_diff = self.neigh_class_pos[:, 6, 0] - self.neigh_class_pos[:, 7, 0]
        self.obs[:, 13] = np.logical_and(y_class_diff == 0, self.neigh_class_pos[:, 6, 0] != -1)
        
        self.obs[:, 14] = np.logical_and(self.obs[:, 12], self.obs[:, 13]) # both lines present
        
        # edge
        self.obs[self.wafer_edge, 15] = 1
        
        # reward update
        error = np.sum(self.classes_pred != self.classes)
        reward = (self.prev_error - error) - self.meas_cost
        self.prev_error = error
        
        if self.boundary_obj:
            done = (np.sum(self.classes_meas[self.boundaries] == -1) == 0)
        else:
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
    
    
def simulator(w, id_num=0, args={}, visual=False, plot_all=False, reward_type=1, data=None): # change default args later
    # unpack args
    if reward_type == 0:
        meas_cost = args["meas_cost"]
    else:
        meas_cost = 0
    nn = args["nn"]
    n_classes = args["n_classes"]
    cell_size = args["cell_size"]
    sp = args["sp"]
    
    env = SpectraSimulator(sp=sp, meas_cost=meas_cost, n_classes=n_classes, test_data=data, cell_size=cell_size)

    done = False
    obs = env.reset()
    measured = np.zeros(obs.shape[0], dtype=bool)
    t = 0
    r = 0
    while not done:
        if nn:
            ranks = nn_policy(w, obs)
        else:
            ranks = obs.dot(w[:-1]) + w[-1]
        
        ranks[measured] = -np.inf
        if np.max(ranks) < 0 and reward_type == 0: done = True
            
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
        if data is None: print(np.where(measured))
      
    if reward_type == 0:
        episode_reward = np.sum(env.classes == env.classes_pred) - t * meas_cost
    else:
        episode_reward = -t
        
    if data is None:
        return id_num, episode_reward
    else:
        return np.array(env.ordered_measurements)

def nn_policy(w, obs):
    device = torch.device('cpu')
    
    model = torch.nn.Sequential(OrderedDict([
        ("lin1", torch.nn.Linear(obs.shape[1], 8)),
        ("relu1", torch.nn.ReLU()),
        ("lin2", torch.nn.Linear(8, 4)),
        ("relu2", torch.nn.ReLU()),
        ("lin3", torch.nn.Linear(4, 1)),
        ("relu3", torch.nn.ReLU()),
    ])).to(device)

    w_used = 0
    for name, param in model.named_parameters():
        param.requires_grad = False # on first usage
        n_used = param.data.numpy().size
        param_shape = param.data.shape
        param.data = torch.tensor(w[w_used:(w_used + n_used)]).reshape(param_shape)
        w_used += n_used

    rank = model(torch.from_numpy(obs)).data.numpy()
    return rank

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
