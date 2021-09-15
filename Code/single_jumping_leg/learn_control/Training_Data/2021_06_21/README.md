# Description of Training Data
This is a short description of the training data that is within this repository. It is meant to act as a history marker for tracking testing of different training parameters including things like hyperparameter selection.

## Model Train Date: 06/18/2021
## Environment: Single Leg Two Link
This is the two linked leg found in the [Single_Leg_Jumping_Robot](https://github.com/CRAWlab/CRAWLAB-Student-Code---2020/tree/master/Andrew%20Albright/Code/single_jumping_leg/single_leg/resources/Single_Leg_Jumping_Robot) repository. This is a simplified example used to learn pybullet.

## Environment Notes:
* Self collision was disabled so the leg could penetrate itself. This is not realistic but again was used as a test.
* Gravity was set to 9.81m/s^2
* Max velocity of the motors was set to 2*pi rad/s
* Max position of the motors was not set so they could spin infinitely

## Training Parameters: 
### Total Steps: 500000
### Number of seeds: 4 (see data)
### Episode Length: 2 seconds (240 x 2 time steps)
