# ALBRIGHT_SEMESTER_PLAN_SPRING_2022

Author: Andrew Albright  
Institution: UL Lafayette  
Date: 01/12/2022

## Thesis Writing Timeline
- [x] 01/25/2022 - Last Day to Apply for Candidacy (Graduate School)
- [x] 02/25/2022 - Last Day to Apply for Graduate Degree
- [x] 01/24/2022 - Outline
- [x] 01/24/2022 - <span style="color:orange"> *Intro Complete* </span>
- [ ] 01/31/2022 - <span style="color:red"> *Chapter 1 Complete* </span> 
  - [x] Re-make the data from the paper
  - [x] Re-Train the single jumping agents
- [x] 02/07/2022 - <span style="color:red"> Chapter 2 Complete </span>
- [x] 02/14/2022 - <span style="color:red"> Chapter 3 Complete </span>
- [x] 02/21/2022 - Chapter 4 Complete
- [x] 02/28/2022 - Chapter 5 Complete
- [x] 03/07/2022 - Pre-Defended Copy of Thesis Completed
- [x] 03/21/2022 - Defended Copy of Thesis Completed
- [ ] 04/04/2022 - Present Defended Copy of Thesis to Committee
- [ ] 04/11/2022 - Last Day to Submit Defended Copy of Theses or Dissertations for Graduate School Editing and Final Approval
- [ ] 04/25/2022 - Last Day to Submit Final Copies of Theses or Dissertations/Last Day to Complete Graduate Written and/or Oral Examinations
- [ ] 04/29/2022 - Last day of classes


## Compliant Robots Research
### Goals to accomplish 
- [ ] 02/14/2022 - Paper: Using Input Shaping to analytically evaluate RL
  - [x] 01/17/2022 - Train Agent:
    - [x] Jump to commanded height
    - [x] Jump w/ varying spring k and zeta
  - 01/21/2022 - Gather the evaluation data
    - [x] How close are we to the specified height?
    - [x] Do we get 1:1 robustness performance where the agents are able to jump according to varied mech params?
  - [x] 10/08/2021 - Generate comparative data
  - [ ] 02/14/2022 - Write Paper
    - [x] 01/28/2022 - Outline
    - [ ] 02/04/2022 - First draft
    - [ ] 02/09/2022 - Mostly written paper

- [ ] 02/21/2022 Paper: Concurrent Design - Pogo Stick
  - [x] 01/28/2022 - *Create gym env:*
  -  [x] 02/04/2022 - Train Agents:
    - Train the agents to jump high/efficiently
    - Train the agents to jump to a certain height
  - [x] 02/11/2022 - Gather Data
  - [ ] 02/21/2022 - Write Paper
    - [ ] 02/09/2022 - Outline
    - [ ] 02/16/2022 - First Draft
    - [ ] 02/18/2022 - Mostly written paper

- [ ] 02/28/2022 - Paper: Two Link Flexible-Leg
  - [x] 02/07/2022 - Solve env issues:
    - The env needs a time off the ground method created
  - [ ] 02/16/2022 - Create concurrent design env
    - Based on the architecture of the pogo concurrent design
  - [ ] 02/21/2022 - Train agents
    - Train the agents to jump high
    - Train the agent to stay off the ground
    - Train the agents to jump to a certain height
    - Train the agents to do all of the above efficiently
      - Do the agents learn effective strategies?
    - Train with mechanical uncertainty, i.e. random range of spring/damper/mass/length values
      - Do the agent learn robust strategies?
  - [ ] 02/23/2022 - evaluate designs in simulation
    - Gather Data
  - Write Paper
    - [ ] 02/16/2022 - <span style="color:red"> Outline </span>
    - [ ] 02/21/2022 - First Draft
    - [ ] 02/28/2022 - Final Draft

- Two Link Flexible Leg - Hardware
  - [x] 01/31/2022 - <span style="color:red"> *Write code to capture leg pose* </span> 
    - [x] 01/24/2022 - Jetson Nano set up
  - [ ] 02/07/2022 - <span style="color:red"> Check code which is capturing feedback from motors </span>
    - [ ] Deploy and capture leg position to verify working product
  - [ ] 02/14/2022 - <span style="color:red"> Write code to interface with hardware and capture data feedback simultaneously </span>
    - [ ] Manual actuation of the leg capturing leg position from the camera and motor position from the motors
  - [ ] 02/21/2022 - All Code written and tested

- [ ] 03/07/2022 - Paper: Sim to Real
  - [ ] 02/21/2022 - Fab leg design based on RL generated design
  - [ ] 02/23/2022 - Perform Sim to Real
  - [ ] 03/07/2022 - Write Paper
    - [ ] 02/23/2022 - Outline
    - [ ] 03/02/2022 - First Draft
    - [ ] 03/07/2022 - Final Draft

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
- [ ] 10/27/2021 - Get a simple GUI put together for controlling motors
  - [ ] On/Off buttons and Speed sliders for:
    - [ ] Clamping System Motor
    - [ ] De-Heading System Motor
  - [ ] Profile Following controller for:
    - [ ] Cutting System Motor
  - [x] Camera Display Screens and Capture Buttons
    - [x] 3 Different Cameras
- [x] 10/29/2021 - Have a Demo in shape for the Board
- [x] 11/05/2021 - Have the Demo perfected
- [ ] DATE - 

### Current Roadblocks
- [ ] Need to get the GUI Webapp working so we can do some better testing.
### Learning Needed
- [ ] N/A

## External Deadlines
* Conference deadlines:
    - 03/01/2022 - [IROS 2022](https://iros2022.org/)
    - 04/08/2022 - [MECC 2022](https://mecc2021.a2c2.org/MECC2022.html)