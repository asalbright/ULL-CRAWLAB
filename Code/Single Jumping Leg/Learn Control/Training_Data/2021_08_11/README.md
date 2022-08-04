# Description of Training Data
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.

## Model Train Date: 08/11/2021
- The training did not finish, only half of it completed. 
- It was multi-processed and only the first half of the training completed
  - Traceback (most recent call last):
  File "training.py", line 151, in <module>
    multiprocess_training(train_agent, TRIAL_SEEDS, 3)
  File "training.py", line 147, in multiprocess_training
    return pool.map(function, i)
  File "/home/crawlab/anaconda3/envs/RL_AA/lib/python3.8/multiprocessing/pool.py", line 364, in map
    return self._map_async(func, iterable, mapstar, chunksize).get()
  File "/home/crawlab/anaconda3/envs/RL_AA/lib/python3.8/multiprocessing/pool.py", line 771, in get
    raise self._value
    
## Environment Notes:
* Self collision was disabled so the leg could penetrate itself. This is not realistic but again was used as a test.
* Gravity was set to 9.81m/s^2

## ENV/TRAINING Parameters: 
### Define the environment ID's and set Environment parameters
env_id = 'SingleLeg-v0'
eval_env_id = env_id
ROBOT_TYPE = "USER_SPECIFIED"
ROBOT_LOCATION = "flexible/basic_flex_jumper/basic_flex_jumper.urdf"
K_P = 0.75
K_V = 1

### Set up the Training parameters
NUM_TRIALS = 6
EPISODE_STEPS = 240*2
TOTAL_STEPS = 1000000
ROLLOUT = 7500
GAMMA = 0.99

### Set up the training seeds
INITIAL_SEED = 70504
EVALUATION_FREQ = 2000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))

### Env Initialization
ROBOT_TYPE,
robotLocation=ROBOT_LOCATION,
evaluating=False,
flexible=False,
epSteps=EPISODE_STEPS,
maxMotorPos=np.deg2rad(30),
maxMotorVel=np.deg2rad(90),
maxMotorForce=100,
positionGain=K_P,
velocityGain=K_V,
maxFlexPosition=np.deg2rad(15),
controlMode="POSITION_CONTROL",
captureData=False,
saveDataName=None,
saveDataLocation=None