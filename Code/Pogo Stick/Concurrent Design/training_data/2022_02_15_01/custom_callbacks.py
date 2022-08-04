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
import numpy as np
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import TD3

from gym_pogo_stick.gym_pogo_stick.envs import PogoStickDesignEnv

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

class TrainingDesignCallback(BaseCallback):
    """
    Custom callback for updating the environment design.
    """

    def __init__(self, train_freq, rew_func, sim_steps, data_name, models_path, log_data=False, verbose=0):
        super(TrainingDesignCallback, self).__init__(verbose)

        self.train_freq = train_freq
        self.called = 0

        self.episode_steps = 1              # Ep steps
        self.sim_step_size = 0.001          # Sim step size
        self.sim_steps = sim_steps           # Sim steps

        # set up the training parameters
        self.num_trials = 10                # Number of agents learning designs
        self.total_sims = 1000              # Number of simulations per agent
        self.rollout = 100                  # Number of random steps per agent
        self.gamma = 0.99                   # Learning Rate

        # Set up the saving locations
        self.data_name = data_name
        if log_data:
            self._create_data_dir()

        # Set up the models path
        self.models_path = models_path

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

        # If the number of times the control model has been trained is divisible by the training frequency
        if self.called % self.train_freq == 0:
            # Save the model to train a design with
            path = self.models_path / f'{self.data_name}_{self.called}'
            self.model.save(path=path)

            # Get the new env design parameters from the design model
            self.design_params = self._train_design(path)

            # Update the design parameters in the control env
            self._update_design_params(self.design_params)
        
    def _train_design(self, path=None) -> Dict:
        # if the path is not specified, use the default
        if path is None:
            path = self.models_path / f'{self.data_name}_{self.called}'

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

        # create the model
        # open tensorboard with the following bash command: tensorboard --logdir ./logs/
        buffer_size = self.total_sims + 1
        model = TD3("MlpPolicy", 
                    design_env, 
                    verbose=self.verbose, 
                    buffer_size=buffer_size, 
                    learning_starts=self.rollout, 
                    tensorboard_log=self.logs_path,
                    seed=trial_seed, 
                    gamma=self.gamma)

        # train the agent
        model.learn(total_timesteps=self.total_sims, 
                    tb_log_name=f'des_{self.data_name}_{int(trial_seed)}'
                    )

        # Get the mech parameters and delete the model and design_env
        params = design_env.unwrapped.params
        design_env.close()
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

    def _create_data_dir(self) -> None:

        self.save_path = Path.cwd()
        self.save_path = self.save_path.joinpath(f'designs_{self.data_name}')
        self.logs_path = self.save_path.joinpath('logs')
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)
        