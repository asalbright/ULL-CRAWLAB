import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
p.setGravity(0,0,-9.81)
p.setPhysicsEngineParameter(enableFileCaching=0)

legId = p.loadSoftBody("Deform_Anchor/CurvedBeamLegWithFeatures.obj", basePosition=[0, 0.1, 0.525], 
                                                                      scale=1/1000, 
                                                                      mass=0.05,
                                                                      useNeoHookean = 0, 
                                                                      useBendingSprings=1,
                                                                      useMassSpring=1, 
                                                                      springElasticStiffness=10000, 
                                                                      springDampingStiffness=25, 
                                                                      springDampingAllDirections = 1, 
                                                                      useSelfCollision = 0, 
                                                                      frictionCoeff = .5, 
                                                                      useFaceContact=1)


# p.createSoftBodyAnchor()
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
baseId = p.loadURDF("Deform_Anchor/single_leg_base/single_leg_base.urdf",cubeStartPos, cubeStartOrientation, 
                    useFixedBase=True,
                    # useMaximalCoordinates=1, ## New feature in Pybullet
                    flags=p.URDF_USE_INERTIA_FROM_FILE)
sliderId = p.loadURDF("Deform_Anchor/single_leg_slider/single_leg_slider.urdf", [0, 0.01, 0.5], cubeStartOrientation,
                    useFixedBase=False,
                    # useMaximalCoordinates=1,
                    flags=p.URDF_USE_INERTIA_FROM_FILE)

sliderJoint = p.createConstraint(parentBodyUniqueId=baseId, parentLinkIndex=0,
                   childBodyUniqueId=sliderId, childLinkIndex=-1,
                   jointType=p.JOINT_PRISMATIC,
                   jointAxis=[0,0,1],
                   parentFramePosition=[0.0,0.04,0.0],
                   childFramePosition=[0.0,0.0,0.0])

p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,1)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)

debug = False
if debug:
  data = p.getMeshData(legId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
#   print("--------------")
#   print("data=",data)
#   print(data[0])
#   print(data[1])
  text_uid = []
  for i in range(data[0]):
      pos = data[1][i]
      uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1])
      text_uid.append(uid)

for i in range (10000):
    if debug:
        data = p.getMeshData(legId, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        for i in range(data[0]):
            pos = data[1][i]
            uid = p.addUserDebugText(str(i), pos, textColorRGB=[1,1,1], replaceItemUniqueId=text_uid[i])

    rKey = ord('r')
    keys = p.getKeyboardEvents()
    if rKey in keys and keys[rKey]&p.KEY_IS_DOWN:
        p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(baseId)
print(cubePos,cubeOrn)
p.disconnect()

