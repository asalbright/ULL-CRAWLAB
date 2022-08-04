################################################################################################################
# File Name: load_env.py
# Author: Andrew Albright 
# 
# Description: file for loading in the .urdf file of the single leg to visualize everything
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
################################################################################################################

import time
import numpy as np
import gym
import pybullet as p
import pybullet_data
from time import sleep
from pathlib import Path
import matplotlib.pyplot as plt

from gym_single_leg.gym_single_leg.envs.single_leg_env import SingleLegEnv
from stable_baselines3.common.env_checker import check_env
from tools.video_functions import VideoWrite

EPISODE_STEPS = 240*12
MOTOR_MAX_POS = np.deg2rad(30)
MOTOR_MAX_VEL = np.deg2rad(330) # 55 RPM -> 330 deg/s
SPRING_K = 0.75
SPRING_DAMPING = 1
FLEX_MAX_POS = np.deg2rad(15)

DEBUG = False
CONTROL_MODE = {"PC": 0, "VC": 1}

def main():
    '''
    Run this file as a script. To exit program and plot data, press the "Close" button
    in the GUI presented. 

    Wait for data to be plotted. The data is collected at every timestep during sim (240Hz).
    The longer the sim is allowed to run, the more data there will be.

    To see changes in solver analytics data, change parameters. 
    '''

    # Declare the env
    env = SingleLegEnv(robotType="USER_SPECIFIED",
                       robot_location="single_leg_sys/single_leg_sys.urdf",
                       show_GUI=True,
                       flexible=True,
                       ep_steps=EPISODE_STEPS,
                       max_motor_pos=MOTOR_MAX_POS,
                       max_motor_vel=MOTOR_MAX_VEL,  # RPM
                       max_motor_force=100,
                       position_gain=SPRING_K,
                       velocity_gain=SPRING_DAMPING,
                       max_flex_pos=FLEX_MAX_POS,
                       save_data=False,
                       saveDataName=None,
                       saveDataLocation=None)

    # check_env(env)

    obs = env.reset()

    # Set debug parameters
    motor_one = p.addUserDebugParameter('Motor1', -MOTOR_MAX_POS, MOTOR_MAX_POS, 0, physicsClientId=env.client)
    motor_two = p.addUserDebugParameter('Motor2', -MOTOR_MAX_POS, MOTOR_MAX_POS, 0, physicsClientId=env.client)
    both_motors = p.addUserDebugParameter('Both Motors', -MOTOR_MAX_POS, MOTOR_MAX_POS, 0, physicsClientId=env.client)
    position_gain = p.addUserDebugParameter('Kp', 0.01, 1.5, 0.75, physicsClientId=env.client)
    velocity_gain = p.addUserDebugParameter('Kv', 0.01, 1, 0.5, physicsClientId=env.client)
    break_simulation = p.addUserDebugParameter('Close', 1, 0, 0, physicsClientId=env.client)
    
    # Lists for storing solver iterations and error data
    num_iterations = []
    error = []
    # Listed for storing flex joint position values
    flex_positions = [[], []]
    flex_velocities = [[], []]
    motor_positions = [[], []]
    motor_velocities = [[], []]
    link_positions = [[], [], []]

    # video_writer = VideoWrite(fps=60, file_name="Single_Leg").start()

    frame = None

    while p.isConnected():
        # Read debug parameters from GUI
        motor_one_input = p.readUserDebugParameter(motor_one, physicsClientId=env.client)
        motor_two_input = p.readUserDebugParameter(motor_two, physicsClientId=env.client)
        both_motors_input = p.readUserDebugParameter(both_motors, physicsClientId=env.client)

        position_gain_input = p.readUserDebugParameter(position_gain, physicsClientId=env.client)
        velocity_gain_input = p.readUserDebugParameter(velocity_gain, physicsClientId=env.client)
        break_simulation_input = p.readUserDebugParameter(break_simulation, physicsClientId=env.client)
        if break_simulation_input: break

        # Set the gains for the flex joints according to the sliders in the GUI
        env.leg.position_gain = position_gain_input
        env.leg.velocity_gain = velocity_gain_input

        # Apply the motor debug parameters to the env motors
        motor_1 = motor_one_input + both_motors_input
        motor_2 = motor_two_input + both_motors_input
        action = motor_1, motor_2
        
        # Apply step in env
        obs, _, done, _ = env.step(action)
        
        link_cartisian_pos, _, _, _, _, _ = env.leg.get_link_state()
        

        # Update the Frame in the video writer
        # if not video_writer.stopped:
        #     video_writer.frame = env.get_camera_image()
        # else: 
        #     video_writer.stop()

        # Update the camera image within the dlc-live process
        # if frame is None:
        #     frame = env.get_camera_image()
        # else:
        #     frame = env.get_camera_image()

        if DEBUG:
            # Get solver analytics from env
            solver_analytics = env.solver_analytics
            # Append solver analytics to the data lists
            num_iterations.append(solver_analytics["numIterationsUsed"])
            error.append(solver_analytics["remainingResidual"])

            # Get the information about the flex joints
            flex_joint_states = env.leg.get_joint_states(env.leg.flex_joints)
            flex_positions[0].append(flex_joint_states[0,0])
            flex_positions[1].append(flex_joint_states[1,0])
            flex_velocities[0].append(flex_joint_states[0,1])
            flex_velocities[1].append(flex_joint_states[1,1])

            # Get the information about the motor joints
            motor_joint_states = env.leg.get_joint_states(env.leg.motor_joints)
            motor_positions[0].append(motor_joint_states[0,0])
            motor_positions[1].append(motor_joint_states[1,0])
            motor_velocities[0].append(motor_joint_states[0,1])
            motor_velocities[1].append(motor_joint_states[1,1])
            
            for cor in range(len(link_positions)):
                link_positions[cor].append(link_cartisian_pos[cor])

        # time.sleep(1/240)

        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

        if done:
            obs = env.reset()
            break
            # time.sleep(1/30)
        
    env.close()
    # video_writer.stop()

    # When the sim is closed, plot the solver analytics data
    if DEBUG:
        sim_time = np.array(range(len(motor_positions[0])))
        sim_time = sim_time / 240

        # num iterations plot
        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Timestep')
        plt.ylabel('Num Iterations')
        plt.plot(list(range(len(num_iterations))), num_iterations, linewidth=2, linestyle='-')
        # solver error plot
        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Timestep')
        plt.ylabel('Residual Error')
        plt.plot(list(range(len(error))), error, linewidth=2, linestyle='-')
        # positions
        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Time (s)')
        plt.ylabel('Motor Positions')
        plt.plot(sim_time, np.rad2deg(motor_positions[0]), linewidth=2, linestyle='-')
        plt.plot(sim_time, np.rad2deg(motor_positions[1]), linewidth=2, linestyle='-')
        # velocities
        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Time (s)')
        plt.ylabel('Flex Positions')
        plt.plot(sim_time, np.rad2deg(flex_positions[0]), linewidth=2, linestyle='-')
        plt.plot(sim_time, np.rad2deg(flex_positions[1]), linewidth=2, linestyle='-')

        # link positions
        fig = plt.figure(figsize=(6,4))
        plt.xlabel('Time (s)')
        plt.ylabel('Slider Coordinate Positions')
        plt.plot(sim_time, link_positions[0], linewidth=2, linestyle='-')
        plt.plot(sim_time, link_positions[1], linewidth=2, linestyle='--')
        plt.plot(sim_time, link_positions[2], linewidth=2, linestyle='-.')

        plt.show()

if __name__ == '__main__':
    main()
    