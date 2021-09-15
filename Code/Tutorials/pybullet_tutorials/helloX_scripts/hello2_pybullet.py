############################################################################
# File Name: hello2_pybullet.py
# Author: Andrew Albright 
# 
# Description: file for learning how to make and utilize a .urdf file
# Notes: https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24
############################################################################

# %% file imports
import pybullet as p    # pybullet
import pybullet_data    # import pybullet envs

# %% set up physics server and define gravity for said server
# Can alternatively pass in p.DIRECT 
client = p.connect(p.GUI)
p.setGravity(0, 0, -9.81, physicsClientId=client)

# %% load in .urdf files from path defined
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
carId = p.loadURDF("racecar/racecar.urdf", basePosition=[0,0,0.5])

# %% set the position of the car based on the cars .urdf base position
position, orientation = p.getBasePositionAndOrientation(carId)

# %% watch the car drop to the ground
for _ in range(10000): 
    p.stepSimulation()
# %% watch the car move forward and to the left
for _ in range(1000): 
    # see documentation for how getBasePositionAndOrientation works
    pos, ori = p.getBasePositionAndOrientation(carId)
    # see documentation for how applyExternalForce works
    p.applyExternalForce(carId, 0, [25, 25, 0], pos, p.WORLD_FRAME)
    p.stepSimulation()
# %%
