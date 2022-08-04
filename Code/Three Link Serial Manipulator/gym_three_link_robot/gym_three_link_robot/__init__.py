###############################################################################
# __init__.py
#
# initialization the gym model for easy use
#
#
# Created: 11/28/2021
#   - Andrew Albright
#   - andrew.albright1@louisiana.edu
#
# Notes:
#    
###############################################################################

from gym.envs.registration import register

register(
    id='three_link_robot_arm-v0',
    entry_point='gym_three_link_robot.gym_three_link_robot.envs:ThreeLinkRobotArmEnv',
)
