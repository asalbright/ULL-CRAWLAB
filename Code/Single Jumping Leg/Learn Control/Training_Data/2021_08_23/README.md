# Description of Training Data
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.

## Model Train Date: 08/23/2021
    
## Environment Notes:
* Self collision was disabled so the leg could penetrate itself. This is not realistic but again was used as a test.
* Gravity was set to 9.81m/s^2

## ENV/TRAINING Parameters: 
### Define the environment ID's and set Environment parameters
```python
ROBOT_TYPE = "USER_SPECIFIED"
ROBOT_LOCATION = "flexible/basic_flex_jumper/basic_flex_jumper.urdf"
EPISODE_STEPS = 240*2
MOTOR_MAX_POS = np.deg2rad(30)
MOTOR_MAX_VEL = np.deg2rad(330) # 55 RPM -> 330 deg/s
FLEX_MAX_POS = np.deg2rad(15)
SPRING_K = 0.75
SPRING_DAMPING = 1
ENV_ID = "SingleLeg-v0"
```

### Set up the Training parameters
```python
NUM_TRIALS = 6
TOTAL_EPISODES = 1000
TOTAL_STEPS = EPISODE_STEPS * TOTAL_EPISODES
ROLLOUT = 7500
GAMMA = 0.99
```

### Set up the training seeds
```python
INITIAL_SEED = 70504
EVALUATION_FREQ = 2000
SAVE_FREQ = 100000
RNG = np.random.default_rng(INITIAL_SEED)
TRIAL_SEEDS = RNG.integers(low=0, high=1000, size=(NUM_TRIALS))
```
### Set up training env
```python
env = SingleLegEnv(robotType=ROBOT_TYPE,
                   robotLocation=ROBOT_LOCATION,
                   showGUI=False,
                   flexible=True,
                   epSteps=EPISODE_STEPS,
                   maxMotorPos=MOTOR_MAX_POS,
                   maxMotorVel=MOTOR_MAX_VEL,  # RPM
                   maxMotorForce=100,
                   positionGain=SPRING_K,
                   velocityGain=SPRING_DAMPING,
                   maxFlexPosition=FLEX_MAX_POS,
                   controlMode="POSITION_CONTROL",
                   captureData=False,
                   saveDataName=None,
                   saveDataLocation=None)
```