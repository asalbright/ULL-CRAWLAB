import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('LunarLander-v2')

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e5))
# Save the agent
model.save("Tutorials/Stable Baselines Getting Started/Examples/saves/"+"lunar_lander")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = DQN.load("Tutorials/Stable Baselines Getting Started/Examples/saves/"+"lunar_lander")
# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Enjoy trained agent
obs = env.reset()
for i in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()