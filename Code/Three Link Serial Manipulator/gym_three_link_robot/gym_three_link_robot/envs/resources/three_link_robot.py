##############################################################
# flexible_two_link.py
# 
# OpenAI gym class for a flexile two-link serial manipulator
#
# Author: Andrew Albright
# Date: 11/28/2021
#
##############################################################

import os
import math
import numpy as np
from scipy.integrate import solve_ivp
import sys

class ThreeLinkSerialManipulator():
    def __init__(self, limitTheta=False, limitThetaDot=True):
        # Set up the initial conditions for the solver
        self.theta1_init = 15.0 * np.pi/180  # Initial angle (rad)
        self.theta1_dot_init = 0.0           # Initial angular velocity (rad/s)
        self.theta2_init = 135.0 * np.pi/180  # Initial angle (rad)
        self.theta2_dot_init = 0.0           # Initial angular velocity (rad/s)
        self.theta3_init = -135.0 * np.pi/180  # Initial angle (rad)
        self.theta3_dot_init = 0.0           # Initial angular velocity (rad/s)

        # Pack the initial conditions into an array
        self.x0 = np.array([self.theta1_init, self.theta1_dot_init, 
                            self.theta2_init, self.theta2_dot_init, 
                            self.theta3_init, self.theta3_dot_init])

        # Set the max torque that can be applied
        self.tau_1_max = 200.0
        self.tau_2_max = 200.0
        self.tau_3_max = 100.0
        self.tau_1 = 0.0
        self.tau_2 = 0.0
        self.tau_3 = 0.0
        
        # set the max displacement angles of all the joints
        self.theta1_max = 150 * np.pi/180
        self.theta2_max = 150 * np.pi/180
        self.theta3_max = 150 * np.pi/180
        self.limit_theta = limitTheta

        # Set the max velocity of all the joints
        self.theta1_dot_max = np.pi * 2
        self.theta2_dot_max = np.pi * 2
        self.theta3_dot_max = np.pi * 2
        self.limit_theta_dot = limitThetaDot

        # Define the numerical values for all the system constants
        # and pack them into aan array for passing
        self.m1 = 10.0
        self.m2 = 10.0
        self.m3 = 10.0
        self.l1 = 2.0
        self.l2 = 2.0 
        self.l3 = 2.0
        self.I1 = 1/12 * self.m1 * self.l1**2
        self.I2 = 1/12 * self.m2 * self.l2**2
        self.I3 = 1/12 * self.m3 * self.l3**2
        
        self.p = np.array([self.m1, self.m2, self.m3, 
                           self.l1, self.l2, self.l3, 
                           self.I1, self.I2, self.I3])

        # Create a set of nominal values which can be used to vary the current set of values
        self.p_nom = np.array([self.m1, self.m2, self.m3, 
                               self.l1, self.l2, self.l3, 
                               self.I1, self.I2, self.I3])

        # Create the integration limits and constants
        self.tau = 0.01
        self.t = 0.0

        # Create workspace limits
        x_lower = -0.1
        x_upper = self.l1 + self.l2 + self.l3
        y_lower = -0.1
        y_upper = self.l1 + self.l2 + self.l3
        self.workspace = ((x_lower, x_upper), (y_lower, y_upper)) # limit the arm to the first quadrant

        # Create a power used var
        self.power = 0.0

    def apply_action(self, action):
        # Update the current state to send to the solver
        self.x0 = self.state
        # Get the torques to apply during the solve, make sure they are within the acceptable ranges
        self.tau_1, self.tau_2, self.tau_3 = action

        self.tau_1 = np.clip(self.tau_1, -self.tau_1_max, self.tau_1_max)
        self.tau_2 = np.clip(self.tau_2, -self.tau_2_max, self.tau_2_max)
        self.tau_3 = np.clip(self.tau_3, -self.tau_3_max, self.tau_3_max)

        # Call the ODE solver
        max_step = self.tau * 0.25  # not in use
        resp = solve_ivp(self._eq_of_motion, [self.t, 
                                              self.t+self.tau], 
                                              self.x0, 
                                            #   method="RK45", 
                                              t_eval=[self.t, self.t+self.tau], 
                                              args=(self.p,),
                                            #   max_step=max_step
                                              )
        # Update the system state
        self.state = resp.y[:, -1]

        # Check the joint constraints
        if self.limit_theta:
            self.check_joint_pos_constraints()
        if self.limit_theta_dot:
            self.check_joint_vel_constraints()

        self.t += self.tau
        
        # Update the power used
        self.power_used += self.get_power_used(action)

        return np.array(self.state)

    def get_power_used(self, action):
        # Power (kW) = Torque (N.m) x Speed (RPM) / 9.5488
        step_power = np.abs(action) * np.abs(self.state[1::2]) / 9.5488
        # Add to the total power used
        power_used = np.sum(step_power)

        return power_used

    # TODO: look into setting up joint limits
    def check_joint_pos_constraints(self):
        # Check joint 1 limit
        if self.state[0] > self.theta1_max:
            self.state[0] = self.theta1_max
            # Check if joint 1 is moving past its limit
            if self.state[1] > 0:
                self.state[1] = 0
        elif self.state[0] < -self.theta1_max:
            self.state[0] = -self.theta1_max
            # Check if joint 1 is moving past its limit
            if self.state[1] < 0:
                self.state[1] = 0
        # Check joint 2 limit
        if self.state[2] > self.theta2_max:
            self.state[2] = self.theta2_max
            # Check if joint 2 is moving past its limit
            if self.state[3] > 0:
                self.state[3] = 0
        elif self.state[2] < -self.theta2_max:
            self.state[2] = -self.theta2_max
            # Check if joint 2 is moving past its limit
            if self.state[3] < 0:
                self.state[3] = 0
        # Check joint 3 limit
        if self.state[4] > self.theta3_max:
            self.state[4] = self.theta3_max
            # Check if joint 3 is moving past its limit
            if self.state[5] > 0:
                self.state[5] = 0
        elif self.state[4] < -self.theta3_max:
            self.state[4] = -self.theta3_max
            # Check if joint 3 is moving past its limit
            if self.state[5] < 0:
                self.state[5] = 0
        
    def check_joint_vel_constraints(self):
        # Check joint 1 velocity limit
        if self.state[1] > self.theta1_dot_max:
            self.state[1] = self.theta1_dot_max
        elif self.state[1] < -self.theta1_dot_max:
            self.state[1] = -self.theta1_dot_max
        # Check joint 2 velocity limit
        if self.state[3] > self.theta2_dot_max:
            self.state[3] = self.theta2_dot_max
        elif self.state[3] < -self.theta2_dot_max:
            self.state[3] = -self.theta2_dot_max
        # Check joint 3 velocity limit
        if self.state[5] > self.theta3_dot_max:
            self.state[5] = self.theta3_dot_max
        elif self.state[5] < -self.theta3_dot_max:
            self.state[5] = -self.theta3_dot_max

    # TODO: look into setting up workspace limits    
    def check_space_constraints(self, tau):
        """
        Check the workspace constraints of the system
        """
        tau_1, tau_2, tau_3 = tau

        link1_end_x, link1_end_y, link2_end_x, link2_end_y, link3_end_x, link3_end_y = self.get_link_end_xy()
        
        # if the end of the first link is less than the x lower limit or greater the the y upper limit
        if link1_end_x < self.workspace[0][0] or link1_end_y > self.workspace[1][1]:
            # if the torque is positive, then set it to zero
            if tau_1 > 0:
                tau_1 = 0.0
        # if the end of the the first link is greater than the x upper limit or less than the y lower limit
        if link1_end_x > self.workspace[0][1] or link1_end_y < self.workspace[1][0]:
            # if the torque is negative, then set it to zero
            if tau_1 < 0:
                tau_1 = 0.0

        # if the end of the second link is less than the x lower limit or greater the the y upper limit
        if link2_end_x < self.workspace[0][0] or link2_end_y > self.workspace[1][1]:
            # if the torque on the first link is positive, then set it to zero
            if tau_1 > 0:
                tau_1 = 0.0
            # if the torque on the second link is positive, then set it to zero
            if tau_2 > 0:
                tau_2 = 0.0
        # if the end of the the second link is greater than the x upper limit or less than the y lower limit
        if link2_end_x > self.workspace[0][1] or link2_end_y < self.workspace[1][0]:
            # if the torque on the first link is negative, then set it to zero
            if tau_1 < 0:
                tau_1 = 0.0
            # if the torque on the second link is negative, then set it to zero
            if tau_2 < 0:
                tau_2 = 0.0

        # if the end of the third link is less than the x lower limit or greater the the y upper limit
        if link3_end_x < self.workspace[0][0] or link3_end_y > self.workspace[1][1]:
            # if the torque on the first link is positive, then set it to zero
            if tau_1 > 0:
                tau_1 = 0.0
            # if the torque on the second link is positive, then set it to zero
            if tau_2 > 0:
                tau_2 = 0.0
            # if the torque on the third link is positive, then set it to zero
            if tau_3 > 0:
                tau_3 = 0.0
        # if the end of the the third link is greater than the x upper limit or less than the y lower limit
        if link3_end_x > self.workspace[0][1] or link3_end_y < self.workspace[1][0]:
            # if the torque on the first link is negative, then set it to zero
            if tau_1 < 0:
                tau_1 = 0.0
            # if the torque on the second link is negative, then set it to zero
            if tau_2 < 0:
                tau_2 = 0.0
            # if the torque on the third link is negative, then set it to zero
            if tau_3 < 0:
                tau_3 = 0.0

        return np.array([tau_1, tau_2, tau_3])

    def get_state(self):
        return self.state
    
    def modify_design(self, params=None, pc=0.25):
        """ 
        Modifies the design of the system 

        If params is None, then the design is modified by a random amount
        If params is not None and just the masses and lengths are passed, new inertias are calculated
        If all params are passed, then the design is modified by the passed params        

        Parameters
        ----------
        params : list
            List of parameters to modify
        pc : float
            Percentage change to modify the parameters by if params is not specified

        """
        # Set up low and high range for changing the masses and lengths
        low = self.p_nom[0:5] - pc*self.p_nom[0:5]
        high = self.p_nom[0:5] + pc*self.p_nom[0:5]

        # Modify the masses and lengths by a random amount according to pc, calculate new inertias
        if params is None:
            # Assign the new masses and lengths
            self.p[0:6] = np.random.uniform(low, high, size=(1,6))
            # Calculate the new inertia matrix as I = 1/12 m*l^2
            self.p[6::] = 1/12 * self.p[0:3] * (self.p[3:6])**2
        # Modify the masses and lengths by the passed params, calculate new inertia matrix
        elif len(params) == 6:
            # Assign the new masses and lengths
            self.p[0:6] = params
            # Calculate the new inertia matrix as I = 1/12 m*l^2
            self.p[6::] = 1/12 * self.p[0:3] * (self.p[3:6])**2
        # Modify the masses, lengths, and inertias by the passed params
        elif len(params) == 9:
            self.p = params
        else:
            raise ValueError('Invalid number of parameters passed to modify_design')

    def reset_state(self, random_start=False, mod_design=False, params=None):
        """ 
        Reset the state of the system depending on evaluation or not
        as well as if we want to vary the system's mechanical params
        """
        if not random_start:
            self.state = np.array([self.theta1_init, self.theta1_dot_init, 
                                   self.theta2_init, self.theta2_dot_init, 
                                   self.theta3_init, self.theta3_dot_init])
        # TODO: currently not setting states randomly when training
        else: 
            raise ValueError("Random start not implemented")
            self.state = np.array([self.theta1_init, self.theta1_dot_init, 
                                   self.theta2_init, self.theta2_dot_init, 
                                   self.theta3_init, self.theta3_dot_init])

        # If we want to modify the design, do so
        if mod_design:
            self.modify_design(params=params)

        # Reset the power used
        self.power_used = 0.0
        
        return self.state

    def get_joint_xy(self):
        """
        Returns the x and y positions of the joints
        """

        joint1_x = 0.0
        joint1_y = 0.0
        joint2_x = self.p[3] * np.cos(self.state[0])
        joint2_y = self.p[3] * np.sin(self.state[0])
        joint3_x = joint2_x + self.p[4] * np.cos(self.state[0] + self.state[2])
        joint3_y = joint2_y + self.p[4] * np.sin(self.state[0] + self.state[2])

        return np.array([joint1_x, joint1_y, joint2_x, joint2_y, joint3_x, joint3_y])

    def get_link_end_xy(self):
        """
        Returns the x and y positions of the end of the links
        """
        _, _, link1_end_x, link1_end_y, link2_end_x, link2_end_y = self.get_joint_xy()
        end_effector_x = link2_end_x + self.p[5] * np.cos(self.state[0] + self.state[2] + self.state[4])
        end_effector_y = link2_end_y + self.p[5] * np.sin(self.state[0] + self.state[2] + self.state[4])

        return np.array([link1_end_x, link1_end_y, link2_end_x, link2_end_y, end_effector_x, end_effector_y])

    def _eq_of_motion(self, t, w, p):
        """ Equations of motion for the two link system"""
    
        theta_1, theta_1_dot, theta_2, theta_2_dot, theta_3, theta_3_dot = w
        m_1, m_2, m_3, l_1, l_2, l_3, I_1, I_2, I_3 = p
        
        sys_ode = [theta_1_dot, 
                   theta_2_dot, 
                   theta_3_dot, 
                   (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot)/2 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3))/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)) + self.tau_3)/(I_3 + l_3**2*m_3/4 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))**2/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))) - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot)/2 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3))/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)) + self.tau_3)/(I_3 + l_3**2*m_3/4 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))**2/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))) - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)) + self.tau_1)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2), 
                   (-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot)/2 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3))/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)) + self.tau_3)/(I_3 + l_3**2*m_3/4 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))**2/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))) - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)), 
                   (-m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot)/2 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3))/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))*(-l_1*l_2*m_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot/2 + l_1*l_2*m_2*np.sin(theta_2)*theta_1_dot*theta_2_dot/2 + m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_1_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3) - np.sin(theta_3)*np.cos(theta_2))*(theta_1_dot + theta_2_dot + theta_3_dot)*theta_1_dot)/2 - m_3*(-2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 - (-m_2*(-l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot)/2 - m_3*(-2*l_1*l_2*(theta_1_dot + theta_2_dot)*np.sin(theta_2)*theta_2_dot - 2*l_1*l_2*np.sin(theta_2)*theta_1_dot*theta_2_dot + l_1*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot) + l_1*l_3*(-np.sin(theta_2)*np.cos(theta_3)*theta_2_dot - np.sin(theta_2)*np.cos(theta_3)*theta_3_dot - np.sin(theta_3)*np.cos(theta_2)*theta_2_dot - np.sin(theta_3)*np.cos(theta_2)*theta_3_dot)*theta_1_dot - l_2*l_3*(theta_1_dot + theta_2_dot)*np.sin(theta_3)*theta_3_dot - l_2*l_3*(theta_1_dot + theta_2_dot + theta_3_dot)*np.sin(theta_3)*theta_3_dot)/2 + self.tau_1)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) + self.tau_2)/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)) + self.tau_3)/(I_3 + l_3**2*m_3/4 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2) - (I_3 + m_3*(l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_3 + m_3*(l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)*(I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2))**2/(I_2 + I_3 + l_2**2*m_2/4 + m_3*(2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2 - (I_2 + I_3 + m_2*(l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1*l_2*np.cos(theta_2) + l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)**2/(I_1 + I_2 + I_3 + l_1**2*m_1/4 + m_2*(2*l_1**2 + 2*l_1*l_2*np.cos(theta_2) + l_2**2/2)/2 + m_3*(2*l_1**2 + 4*l_1*l_2*np.cos(theta_2) + 2*l_1*l_3*(-np.sin(theta_2)*np.sin(theta_3) + np.cos(theta_2)*np.cos(theta_3)) + 2*l_2**2 + 2*l_2*l_3*np.cos(theta_3) + l_3**2/2)/2)))]

        return sys_ode