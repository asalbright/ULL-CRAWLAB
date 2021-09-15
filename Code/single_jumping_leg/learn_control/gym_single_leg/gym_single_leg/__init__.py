from gym.envs.registration import register

register(
    id='SingleLeg-v0',
    entry_point='gym_single_leg.envs:SingleLegEnv'
)