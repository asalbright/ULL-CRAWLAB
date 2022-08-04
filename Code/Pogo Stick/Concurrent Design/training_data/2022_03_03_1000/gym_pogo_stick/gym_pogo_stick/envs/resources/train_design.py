from stable_baselines3 import TD3

def train_design(ctrl_model):

    # Set up training env
        design_env = PogoStickDesignEnv(model=ctrl_model,
                                        ep_steps=self.episode_steps,
                                        sim_steps=self.sim_steps, 
                                        reward_function=self.reward_function,
                                        verbose=True)

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
                    # tensorboard_log=self.logs_path,
                    seed=trial_seed, 
                    gamma=self.gamma)

        # train the agent
        model.learn(total_timesteps=self.total_sims, 
                    # tb_log_name=f'des_{self.data_name}_{int(trial_seed)}'
                    )

        # Get the mech parameters and delete the model and design_env
        params = design_env.unwrapped.params
        design_env.close()
        del model

        # TODO: Check that these are the latest ones
        return params

if __name__ == "__main__":

    # Set up training env
    design_env = PogoStickDesignEnv(model=self.model,
                                    ep_steps=self.episode_steps,
                                    sim_steps=self.sim_steps, 
                                    reward_function=self.reward_function,
                                    verbose=True)

    model = TD3.load(model)