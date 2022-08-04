#!/usr/bin/env python

##########################################################
# optimize_design.py
# Author: Andrew Albright
# Date: 11/30/2021
#
# Description:
#   This script is used to optimize the design for the 
#   given three-link serial manipulator.
##########################################################

import gym
import numpy as np
import pandas as pd 
import datetime
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from stable_baselines3 import TD3

from scipy.optimize import Bounds, minimize, LinearConstraint

from gym_three_link_robot.gym_three_link_robot.envs import ThreeLinkRobotArmEnv
from functions import getFiles, getFileNames, readCSVFiles, parseDataFrame, queueDirectoryPath, combineDataInFolder, dfAverageStd, guiYesNo

CAPTURE_DATA = False
EP_STEPS = 500

class EvaluateDesign():
    def __init__(self, env_id, model_id, model_path, save_path, cap_env_data=False):

        self.save_path = save_path

        self._load_environment(env_id, cap_env_data)
        self._load_model(model_id, model_path)

    def evaluate_agent(self, params, render=False):
        # Get the targets the agent will attempt to reach
        targets = [(4, 4), (3, 3)]
        # targets = self.genetate_targets()

        # Set the power used to reach the targets to zero
        power_used = 0.0
        for target in targets:
            self.env.specified_pos = target
                
            # Evaluate the agent
            done, state = False, None
            obs = self.env.reset()
            # Set the environment parameters
            self.modify_env_design(params)

            while not done:
                action, state = self.model.predict(obs, state=state, deterministic=True)
                obs, reward, done, info = self.env.step(action)

                if render:
                    self.env.render()

            # Update the power used to reach the target
            power_used += self.env.robot.power_used

            self.env.close()

        # print(f"Power used with {self.env.robot.p}: {power_used}")

        return float(power_used)

    def modify_env_design(self, params):
        # Modify the environment parameters
        self.env.robot.modify_design(params)

    def _load_environment(self, env_id, cap_env_data):
        # Create and wrap the environment
            self.env = gym.make(env_id)
            self.env.capture_data = CAPTURE_DATA
            self.env.data_location = self.save_path
            self.env.ep_steps = 65

            # Assign the values of the nominal length of the links for use outside the class
            self.p_nom = self.env.robot.p_nom
            
    def _load_model(self, model_id, model_path):
        # Load the model
        self.model = TD3.load(path=model_path / model_id)
    

    def genetate_targets(self, range=[1, 4.2]):
        xy = np.linspace(range[0], range[1], num=3)
        targets = [(xy[0],xy[0]), (xy[0],xy[1]), (xy[0],xy[2]), 
                (xy[1],xy[0]), (xy[1],xy[1]), (xy[1],xy[2]), 
                (xy[2],xy[0]), (xy[2],xy[1]), (xy[2],xy[2])]
        return targets

def optimize():
    env_id = "three_link_robot_arm-v0"

    model_id = Path("agent_12345_900000_steps.zip")
    model_path = Path("Training_Data/TD3/NonLimited_NonRandom/models")

    # model_id = Path("final_agent_7671.zip")
    # model_path = Path("Training_Data/TD3/Limited_NonRandom/models")
    
    eval_design = EvaluateDesign(env_id=env_id, 
                                 model_id=model_id, 
                                 model_path=model_path, 
                                 save_path="Eval_Data/")

    # Function to optimize
    params = eval_design.p_nom[0:6]
    params = list(params)

    # Set the function
    fun = eval_design.evaluate_agent
    
    # test = fun(params, render=True)

    # Constraints
    # m1 + m2 + m3 + 0l1 + 0l2 + 0l3 = m1_nom + m2_nom + m3_nom
    # 0m1 + 0m2 + 0m3 + l1 + l2 + l3 = l1_nom + l2_nom + l3_nom
    rhs_m = eval_design.p_nom[0] + eval_design.p_nom[1] + eval_design.p_nom[2]
    rhs_l = eval_design.p_nom[3] + eval_design.p_nom[4] + eval_design.p_nom[5]
    lb = [rhs_m]
    ub = [rhs_m]
    ar = [1, 1, 1, 0, 0, 0]

    linear_constraint1 = LinearConstraint(ar, lb, ub)

    lb = [rhs_l]
    ub = [rhs_l]
    ar = [0, 0, 0, 1, 1, 1]
    linear_constraint2 = LinearConstraint(ar, lb, ub)


    # Define bounds
    lower_bound = 0.75 * eval_design.p_nom[0:6]
    upper_bound = 1.25 * eval_design.p_nom[0:6]
    bounds = Bounds(lower_bound, upper_bound)

    # resp = minimize(fun, params, method='SLSQP', args=False, bounds=bounds, constraints=[linear_constraint1, linear_constraint2],
    #                 options={'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None})
    resp = minimize(fun, params, args=False, method='trust-constr', hess=None, hessp=None, bounds=bounds, constraints=[linear_constraint1, linear_constraint2], tol=None, callback=None, options={'xtol': 1e-08, 'gtol': 1e-08, 'barrier_tol': 1e-08, 'sparse_jacobian': None, 'maxiter': 1000, 'verbose': 2, 'finite_diff_rel_step': None, 'initial_constr_penalty': 1.0, 'initial_tr_radius': 1.0, 'initial_barrier_parameter': 0.1, 'initial_barrier_tolerance': 0.1, 'factorization_method': None, 'disp': False})
    return resp

if __name__ == "__main__":

    # Optimize the design
    res = optimize()
    print(res)