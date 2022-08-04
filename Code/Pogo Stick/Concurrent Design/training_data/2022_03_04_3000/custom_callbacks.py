#! /usr/bin/env python

###############################################################################
# custom_callbacks.py
#
# Callbacks to use during training
#
# Created: 01/04/2022
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
###############################################################################

import os
import sys
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickDesignEnv
from functions import *

class ControllerLogCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self):
        super(ControllerLogCallback, self).__init__()
        
        self.heights = []   # This will be a list of lists of heights
        self.eposide = 0   # Set to -1 so that the first episode will be 0
        self.params = None  

    def _on_step(self) -> bool:
        self._capture_data()
        return True

    def _capture_data(self) -> None:
        # Capture the height at the current timestep
        env = self.training_env.envs[0].env
        state = env.state
        height = env.state[0]

        # Add a new list for every episode
        if env.timestep == 1:
            self.heights.append([])
            self.eposide += 1
        
        # Append the lastest height to the current eposides list
        self.heights[self.eposide - 1].append(height)   # -1 so that the first episode will be 0
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Log the max height of the episodes
        self._log_max_height()
        # Log the mech parameters of the episode
        self._log_mech_params()


    def _log_max_height(self) -> None:
        """
        Logs the max height of the system
        """
        max_height = np.max(self.heights)
        self.logger.record('max_height', max_height)

        self.heights = []
        self.eposide = -1

    def _log_mech_params(self) -> None:
        """
        Logs the mech parameters
        """
        # Get the mech parameters of the last episode
        self.params = self.model.env.envs[0].env.pogo_stick.get_params()

        self.logger.record('des_spring_k', self.params['k'])
        self.logger.record('des_zeta', self.params['zeta'])

class LogMechParamsCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
        """
        def __init__(self, verbose=0):
            super(LogMechParamsCallback, self).__init__(verbose)

            self.spring_ks = []
            self.spring_cs = []
            self.zetas = []

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass

        def _on_rollout_start(self) -> None:
            """
            A rollout is the collection of environment interaction
            using the current policy.
            This event is triggered before collecting new samples.
            """
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: (bool) If the callback returns False, training is aborted early.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            # Log the mechanical parameters chosen
            # FIXME: This is not working
            params = self.training_env.actions[0]
            self.logger.record('des_spring_k', params[0])
            self.logger.record('des_zeta', params[1])

            self.spring_ks.append(params[0])
            self.zetas.append(params[1])

        def _log_mech_params(self):
            pass

        def _on_training_end(self) -> None:
            """
            This event is triggered before exiting the `learn()` method.
            """
            # TODO: Save the mechanical parameters to a file
            pass

        def _save_mech_params(self):
            """ 
            Save the mechanical parameters to a file
            """
            pass

class BaseDesignCallback(BaseCallback):
    """
    Custom callback for updating the environment design.
    """

    def __init__(self, train_freq, rew_func, sim_steps, data_name, model_path, learn_steps=500, verbose=0):
        super(BaseDesignCallback, self).__init__(verbose)

        self.train_freq = train_freq
        self.called = 0

        self.episode_steps = 1              # Ep steps
        self.sim_step_size = 0.001          # Sim step size
        self.sim_steps = sim_steps           # Sim steps

        # set up the training parameters
        self.learn_steps = learn_steps              # Number of simulations per agent
        self.rollout = int(self.learn_steps * 0.05) # Number of random steps per agent
        self.gamma = 0.99                           # Learning Rate

        # Set up the saving locations
        self.data_name = data_name
        self._create_data_dir()
        self.des_model_name = 'des_model'
        self.des_buffer_name = 'des_buffer'

        # Get the path where the controller model is saved
        self.ctrl_model_path = model_path

        # Set up the reward function
        self.reward_function = rew_func

        # Verbosity
        self.verbose = verbose

        # Attribute for saving the design params found
        self.design_params = None


    def _on_step(self) -> bool:        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # Increment the number of times the control model has been trained
        self.called += 1
        if self.verbose: print(f'Controller has been trained {self.called} times')

        # If the number of times the control model has been trained is divisible by the training frequency
        if self.called % self.train_freq == 0:
            # Save the model to train a design with
            path = self.ctrl_model_path / f'{self.data_name}_{self.called}'
            self.model.save(path=path)

            # Get the new env design parameters from the design model
            self.design_params = self._train_design(path)

            # Update the design parameters in the control env
            self._update_design_params(self.design_params)
        
    def _train_design(self, path=None) -> Dict:
        # if the path is not specified, use the default
        if path is None:
            path = self.ctrl_model_path / f'{self.data_name}_{self.called}'

        # use the path to load the model
        model = TD3.load(path=path)

        # Set up training env
        design_env = PogoStickDesignEnv(model=model,
                                        ep_steps=self.episode_steps,
                                        sim_steps=self.sim_steps, 
                                        reward_function=self.reward_function,
                                        verbose=False)

        # set the trial seed for use during training
        trial_seed = int(self.model.seed)

        design_env.seed(seed=self.model.seed)

        # wrap the design_env in modified monitor which plots to tensorboard the jumpheight
        design_env = Monitor(design_env)

        try:
            # load the model and the replay buffer
            path = self.des_model_path / self.des_model_name
            model = TD3.load(path=path, env=design_env)
            path = self.des_model_path / self.des_buffer_name
            model.load_replay_buffer(path=path)

        except:
            # Make the model
            buffer_size = self.learn_steps + 1
            model = TD3("MlpPolicy", 
                        design_env, 
                        verbose=self.verbose, 
                        buffer_size=buffer_size, 
                        learning_starts=self.rollout, 
                        tensorboard_log=self.des_log_path,
                        seed=trial_seed, 
                        gamma=self.gamma)

        # train the agent
        model.learn(total_timesteps=self.learn_steps, 
                    tb_log_name=f'des_{self.data_name}_{int(trial_seed)}'
                    )

        # Get the mech parameters and close the design env
        params = design_env.unwrapped.params
        design_env.close()

        # Save the design model and the replay buffer and delete the model
        path = self.des_model_path / self.des_model_name
        model.save(path=path)
        path = self.des_model_path / self.des_buffer_name
        model.save_replay_buffer(path=path)
        del model

        # TODO: Check that these are the latest ones
        return params

    def _update_design_params(self, params) -> None:
        """
        Update the design parameters for the control env
        """
        # TODO: make sure the env is not resetting and changing these back to the normal values
        print(f'Updated design params: {params}')
        self.model.env.envs[0].env.params = params

    def get_avg_dict(self, list):
        '''
        This function calculates the average value of each key from a list of dictionaries.

        Parameters
        ----------
        dict : list
            The list if dictionaries to calculate the average value of each key

        Returns
        -------
        dict
            The dictionary with the average value of each key
        '''
        
        avg_dict = {}
        for d in list:
            for key in d.keys():
                if key in avg_dict.keys():
                    avg_dict[key].append(d[key])
                else:
                    avg_dict[key] = [d[key]]
        
        for key in avg_dict.keys():
            avg_dict[key] = np.mean(avg_dict[key])

        return avg_dict

    def _create_data_dir(self) -> None:

        self.save_path = Path.cwd()
        self.save_path = self.save_path.joinpath(f'des_{self.data_name}')
        self.des_log_path = self.save_path.joinpath('logs')
        self.des_model_path = self.save_path.joinpath('models')
        if not os.path.exists(self.des_log_path):
            os.makedirs(self.des_log_path)
        if not os.path.exists(self.des_model_path):
            os.makedirs(self.des_model_path)

class TrainingDesignContinuousCallback(BaseDesignCallback):
    """
    Custom callback for updating the environment design using a continuous design model. 

    That is throughout the training process of the controller, the same design model is used to update the environment.

    Parameters
    ----------
    model_path : str
        The path to the control model to use for updating the environment design
    data_name : str
        The name of the data to save the model and replay buffer to
    train_freq : int
        The number of times the control model has been trained before updating the environment design
    rollout : int
        The number of rollouts to use to train the design model
    sim_steps : int
        The number of simulation steps to use for training the design model
    episode_steps : int
        The number of steps to run the design model for
    reward_function : str
        The reward function to use for training the design model
    verbose : bool
        Whether to print the training progress
    """

    def __init__(self, train_freq, rew_func, sim_steps, data_name, model_path, learn_steps=500, num_models=5, verbose=0):
        super(TrainingDesignContinuousCallback, self).__init__(train_freq, rew_func, sim_steps, data_name, model_path, learn_steps, verbose)
        self.num_models = num_models
        
    def _train_design(self, path=None) -> Dict:
        # If the path is not specified, use the default
        if path is None:
            path = self.ctrl_model_path / f'{self.data_name}_{self.called}'

        # use the path to load the model
        ctrl_model = TD3.load(path=path)

        # set the trial seed for use during training
        ctrl_seed = int(self.model.seed)

        # Get list in initialization seeds
        rng = np.random.default_rng(ctrl_seed)
        trial_seeds = rng.integers(low=0, high=10000, size=(self.num_models))

        # Create an empty list to store the learned params
        params_list = []
        # For every trial in trial_seeds, train the model
        for seed in trial_seeds:

            # Set up training env
            design_env = PogoStickDesignEnv(model=ctrl_model,
                                            ep_steps=self.episode_steps,
                                            sim_steps=self.sim_steps, 
                                            reward_function=self.reward_function,
                                            verbose=False)

            

            design_env.seed(seed=int(seed))

            # wrap the design_env in modified monitor which plots to tensorboard the jumpheight
            design_env = Monitor(design_env)

            try:
                # load the model and the replay buffer
                path = self.des_model_path / f"{int(ctrl_seed)}" / f"{self.des_model_name}_{int(seed)}"
                model = TD3.load(path=path, env=design_env)
                path = self.des_model_path / f"{int(ctrl_seed)}" / f"{self.des_buffer_name}_{int(seed)}"
                model.load_replay_buffer(path=path)

            except:
                # Make the model
                buffer_size = self.learn_steps + 1
                logs_path = self.des_log_path / f"ctrl_{ctrl_seed}"
                model = TD3("MlpPolicy", 
                            design_env, 
                            verbose=self.verbose, 
                            buffer_size=buffer_size, 
                            learning_starts=self.rollout, 
                            tensorboard_log=logs_path,
                            seed=int(seed), 
                            gamma=self.gamma)

            # train the agent
            model.learn(total_timesteps=self.learn_steps, 
                        tb_log_name=f'des_{int(seed)}'
                        )

            # Get the mech parameters and close the design env
            params = design_env.unwrapped.params
            params_list.append(params)
            design_env.close()

            # Save the design model and the replay buffer and delete the model
            path = self.des_model_path / f"{int(ctrl_seed)}" / f"{self.des_model_name}_{int(seed)}"
            model.save(path=path)
            path = self.des_model_path / f"{int(ctrl_seed)}" / f"{self.des_buffer_name}_{int(seed)}"
            model.save_replay_buffer(path=path)
            del model

        # TODO: Check that these are the latest ones
        return params

class TrainingDesignDescreteCallback(BaseDesignCallback):
    """
    Custom callback for updating the environment design using a descrete design model. 

    That is throughout the training process of the controller, a new design model is used to update the environment every update.

    Parameters
    ----------
    model_path : str
        The path to the control model to use for updating the environment design
    data_name : str
        The name of the data to save the model and replay buffer to
    train_freq : int
        The number of times the control model has been trained before updating the environment design
    rollout : int
        The number of rollouts to use to train the design model
    sim_steps : int
        The number of simulation steps to use for training the design model
    episode_steps : int
        The number of steps to run the design model for
    reward_function : str
        The reward function to use for training the design model
    verbose : bool
        Whether to print the training progress
    """

    def __init__(self, train_freq, rew_func, sim_steps, data_name, model_path, learn_steps=500, num_models=5, verbose=0):
        super(TrainingDesignDescreteCallback, self).__init__(train_freq, rew_func, sim_steps, data_name, model_path, learn_steps, verbose)
        
        self.num_models = num_models

    def _train_design(self, path=None) -> Dict:
        # if the path is not specified, use the default
        if path is None:
            path = self.ctrl_model_path / f'{self.data_name}_{self.called}'

        # Load the control model to simulate the design env
        ctrl_model = TD3.load(path=path)

        # Set the trial seed for use during training
        ctrl_seed = int(self.model.seed)
        
        # Get list in initialization seeds
        rng = np.random.default_rng(ctrl_seed)
        trial_seeds = rng.integers(low=0, high=10000, size=(self.num_models))

        # Create an empty list to store the learned params
        params_list = []
        # For every trial in trial_seeds, train the model
        for seed in trial_seeds:
            
            # Set up training env
            design_env = PogoStickDesignEnv(model=ctrl_model,
                                            ep_steps=self.episode_steps,
                                            sim_steps=self.sim_steps, 
                                            reward_function=self.reward_function,
                                            verbose=False)

            design_env.seed(seed=int(seed))

            # Wrap the design_env in modified monitor which plots to tensorboard the jumpheight
            design_env = Monitor(design_env)


            

            # Make the model
            buffer_size = self.learn_steps + 1
            logs_path = self.des_log_path / f"ctrl_{ctrl_seed}"
            model = TD3("MlpPolicy", 
                        design_env, 
                        verbose=self.verbose, 
                        buffer_size=buffer_size, 
                        learning_starts=self.rollout, 
                        tensorboard_log=logs_path,
                        seed=int(seed), 
                        gamma=self.gamma)

            # Train the agent
            model.learn(total_timesteps=self.learn_steps, 
                        tb_log_name=f'des_{int(seed)}'
                        )

            # Get the mech parameters and close the design env
            params = design_env.unwrapped.params
            params_list.append(params)
            design_env.close()

            # Delete the model
            del model

        # Find the average of the params
        params = self.get_avg_dict(list=params_list)

        # TODO: Check that these are the latest ones
        return params