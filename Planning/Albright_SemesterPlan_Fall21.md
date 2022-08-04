# ALBRIGHT_SEMESTER_PLAN_FALL_2021

Author: Andrew Albright  
Institution: UL Lafayette  
Date: 09/13/2021

## Compliant Robots Research
### Goals to accomplish 
- 09/28/2021 - ACC Paper Requirements 
  - [x] 09/15/2021 - Modify Environment: Create a stop at maximum spring deflection.
  - [x] 09/17/2021 - Train Agent: Gather the data which is the learning of the mechanical design parameters.
  - [x] 09/20/2021 - Make plots for paper
  - [x] 09/21/2021 - Finish Paper: Write the remaining sections pertaining to the data collection.
- 11/15/2021 - Paper: Using Input Shaping to analytically evaluate RL
  - 09/29/2021 - Train Agent:
    - [x] Jump to commanded height
    - Jump w/ varying spring k and zeta
  - 10/01/2021 - Gather the evaluation data
    - [x] How close are we to the specified height?
    - Do we get 1:1 robustness performance where the agents are able to jump according to varied mech params?
  - 10/08/2021 - Generate comparitave data
  - 10/22/2021 - Write Paper
    - 10/06/2021 - Outline
    - 10/15/2021 - First draft
    - 10/21/2021 - Mostly written paper
- 09/29/2021 Concurrent Design - Pogo Stick
  - [x] 09/21/2021 - Create gym env:
    - [x] Based on learned practices from the pogo env
  -  09/24/2021 -Train Agents:
    - Train the agents to jump high
    - Train the agent to stay off the ground
    - Train the agents to jump to a certian height
    - Train the agents to do all of the above efficiently
      - Do the agents learn effective strategies?
  - 09/28/2021 - Gather Data
- Two Link Flexibe Leg - RL
  - 09/16/2021 - Solve env issues:
    - The eval env is having issues, see Running/Jumping TODO 08_09_2021
  - 10/01/2021 - Create concurrent design env
    - Based on the arcitecture of the pogo concurrent design
  - 10/06/2021 - Train agents
    - Train the agents to jump high
    - Train the agent to stay off the ground
    - Train the agents to jump to a certian height
    - Train the agents to do all of the above efficiently
      - Do the agents learn effective strategies?
    - Train with mechanical uncertianty, i.e. random range of spring/damper/mass/length values
      - Do the agent learn robust strategies?
  - 10/08/2021 - evaluate designs in simulation
    - Gather Data
- Two Link Flexible Leg - Hardware
  - 10/10/2021 - Write code to capture leg pose
  - 10/13/2021 - Check code which is capturing feedback from motors
  - 10/15/2021 - Write code to interface with hardware and caputure data feedback simultaniuously
    - ROS would be helpful I am sure
- 11/15/2021 - Paper: Concurrent Design of a Flexible Legged Jumping System
  - 10/08/2021 - Gather Data
  - 10/20/2021 - Outline
  - 10/25/2021 - First Draft
  - 10/29/2021 - Final Draft
- 03/01/2022 - IROS 2022 - Paper: Sim to Real
  - 11/19/2021 - Fab leg design based on RL generated design
  - 11/23/2021 - Evaluate base performance of agent pre-training
  - 11/30/2021 - Perform post training on hardware

### Current Roadblocks
- Need to look more into the sim to real aspect
  - Not sure about the sample time from the hardware and if it needs to match the sim sample time
- If RL is a suitable resource for designing agents hardware aspects
  - Might need to look towards ES
### Learning Needed
- RL sim to real
- Learn more about the application of ES for mechanical design of robotics systems for task specific applications


## Crawfish Peeler Team - Spring 2021
### Goals to accomplish 
- [x] 10/06/2021 - Complete clamp design
  - [x] 09/27/2021 - Test different design sized of current design
  - [x] Does it work?
    - [x] Yes:
      - [x] 09/30/2021 - Design the cam which will actuate it
      - [x] 10/04/2021 - Integrate it into the system
    - NO:
      - 10/06/2021 - Consider a redesign. We alrady have some ideas in case.
- [x] 09/24/2021 - Get timing switch mounted
  - Ping DV when we need to wire it up
- [x] 09/29/2021 - Cutting System assembled
- [x] 10/13/2021 - Conrol/Testing of cutting system.
- [x] 10/20/2021 - Test all systems and gather perfromance data 
  - [x] Cutting system pending the arrival of the blades
- [x] 10/27/2021 - Consider changing the extraction to a motor actuated system.
  - [x] More expensive but more reliable potentially. 
  - [x] If the design was less plastic and 3D printed it would also be more mechanically sound and smooth.
  - [x] If we redesign, there would be no need to worry about performance issues. It would work in the same way, just smoother and in a more dynamic way.
- 10/27/2021 - Get a simple GUI put together for controlling motors
  - On/Off buttons and Speed sliders for:
    - Clamping System Motor
    - De-Heading System Motor
  - Profile Following controller for:
    - Cutting System Motor
  - [x] Camera Display Screens and Capture Buttons
    - [x] 3 Different Cameras
- [x] 10/29/2021 - Have a Demo in shape for the Board
- [x] 11/05/2021 - Have the Demo perfected

### Current Roadblocks
- [x] Need to get the clamp design working properly
- Need to get a simple GUI for testing and demoing

### Learning Needed
- N/A

## External Deadlines
* Conference deadlines:
    - 09/28/2021 - [ACC 2021](https://acc2022.a2c2.org/) 
    - 09/28/2021 - [2022 ARCI](https://arci-conference.com/)
    - 03/01/2022 - [IROS 2022](https://iros2022.org/)