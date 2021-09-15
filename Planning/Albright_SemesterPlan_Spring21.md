# ALBRIGHT_SEMESTER_PLAN_SPRING_2021

Author: Andrew Albright  
Institution: UL Lafayette  
Date: 02/12/2021 

## Research Plan - Spring 2021

Idea: Reinforcement Learning and Evolutionary Strategies can be used to design the mechanical aspects of a non-rigid single legged jumping system to improve performance. Proof of improvement is seen when the system can accomplish a task close to identical as one achieved on a rigid-system while conserving power.

### Goals to accomplish 

* **CODE WRITING** - Define the Reinforcement Learning (RL), Evolutionary Strategies (ES) and ROS code. This code will be utilized to create the mechanical design of the flexible leg structure, the control algorithm and the sim-to-real integration of the system.
    * **Reinforcement Learning code** - will be used to develop the control path for making the single-legged system jump. Height and Power Conservation will be the rewarding functions.
      * Define the algorithm(s) to be used.
      * Get single-leg OpenAI type environment operating with algorithm
        * Choose a leg design and get the control path defined through training
      * Connect RL aspect to ES aspect
      * Get RL aspect training on iterations of ES defined legs
    * **Evolutionary Strategies code** - will be used to develop the mechanical aspects of the single-legged system. The things we will be looking to change will in relation to links (length, width, thickness and infill)
      * Get ES to iteratively define bounded leg structures for RL algorithm to train a control path for
      * Connect ES to RL to create the iterative process
    * **ROS code** - will be used to develop visual simulations as well as assist in the sim-to-real process of the project.
      * Get rigid single-leg system running in ROS
      * Simulate simple jumps to gather data (see GATHER DATA TO EVALUATE PERFORMANCE for more info).


* **HARDWARE DESIGNED** - Develop 10 different rigid and flexible leg designs to be used to evaluate the performance of the single legged system.
  * **Rigid System - Regular Control Strategy** - to be used to test regular control strategies to evaluate performance of system and test hardware.
    * Develop 10 different designs taking into consideration the weight of the design as this will be a factor of power savings.
    * Create models of the designs in Solidworks.
  * **Flexible System - Regular Control Strategy** - to be used to compare the Rigid and Flexible (regular control strategy) designs. Also to be used to compare human designed and controlled flexible systems to the AI designed and controlled flexible systems.
    * Develop 10 different designs taking into consideration the bounds that will be put on the ES aspect of the AI algorithm as to allow for a fair comparison.
    * Create models of the designs in Solidworks.
  * **Flexible System - AI Control Strategy** - defined by developed AI algorithm, these will be used to evaluate the viability of our theory.
    * Convert 10 different AI defined parts to Solidworks designs.


* **HARDWARE FABRICATED/PURCHASED & TESTED** - Servos, links (legs), controller boards, wiring harnesses, system frame, camera and sensors (tbd) need to be gathered and tested. These will be used for both evaluating the performance on various rigid designs and the RL/ES defined links.
  * **Gather Hardware** - 
    * Parts list here.
  * **Test Hardware** - Test the hardware utilizing the time to gather data on several rigid and flexible system designs. See GATHER DATA TO EVALUATE PERFORMANCE for more information.
    * subtask
  * **Fabricate Hardware** - Fabricate both rigid and non-rigid designs using 3D printing techniques.
    * **Rigid System - Regular Control Strategy** 
      * 3D print models.
    * **Flexible System - Regular Control Strategy**
      * 3D print models.
    * **Flexible System - AI Control Strategy**
      * 3D Print models.


* **GATHER DATA TO EVALUATE PERFORMANCE** - Collect data using hardware and software defined, which proves the viability of using an ES-RL strategy to define both mechanical and control algorithm aspects of the single-legged system for the purposes of conserving power. 
  * **Rigid System - Regular Control Strategy** - Collect data using 10 different rigid system designs. This will be used to evaluate the differences seen in different mechanical designs of the system. This will also be used as a benchmark for comparing rigid and flexible systems.
    * Assemble Rigid System links into frame.
    * 
  *  **Flexible System - Regular Control Strategy** - Collect data using 10 different flexible system designs. This will be used to evaluate the differences seen between the Rigid and Flexible (regular control strategy) designs. This will also be used to see the differences between flexible designs which are human designs and controlled and ones which are developed and controlled by out AI algorithm.
     * Assemble Flexible System links into frame.
  * **Flexible System - ES-RL Control Strategy** - Collect data on 10 different optimal designs defined by the developed AI algorithm. This will be used to prove that a ES-RL developed system has the potential to define mechanical and control path aspects of the single-legged system while also achieving more power conservation than traditional design methods.
    * Perform sim-to-real tasks.
      * subtask
    * subtask

* **DEVELOP TWO PAPERS** - Ideas so far
    * **RL for Training Power Conservative Simulated Flexible Robots**
    * **RL for Training Higher Jumping Simulated Flexible Robots** 
    * Same as above but focusing on results seen from actual hardware

### Timeline of Goals - Needs updating based on **GOALS TO ACCOMPLISH**
* 01/16 - 02/15
    - Working with team to become familiar with software environment 
    - running simulations to evaluate performance feasibility
    - document process to write paper on simulation performance
* 02/16 - 03/15
    - perform and evaluate sim-to-real process
    - document for writing paper on sim-to-real feasibility
    - finish paper on simulation performance
* 03/16 - 04/30
    - begin process of complicating the design of the robot
    - perform simulation of complicated design
    - finish paper on sim-to-real feasibility

### Current Roadblocks
* Simulation experience - I do not have a lot of experience getting "hands on" utilizing OpenAI and ROS.
* The tasks ahead, as I see them, simply put are:
    - OpenAI to describe the robot and create a port to use RL to solve the problem
      - I have some experience, but limited
      - My experience with ES is more limited, too
    - ROS to simulate solving and solved problem and to provide the port for the sim-to-real process
      - essentially no experience
      - lots of reading and watching videos

### Learning Needed
* I need to learn more about OpenAI environments in general. To do this, I am going to need to get hands on and learn by doing and asking questions. This is how I learn best.
* I also need to learn more about ROS. To do this I will need to do the same as OpenAI.

### External Deadlines
* Conference deadlines:
    - [AIM 2021](https://aim2021.org/important-dates/) - February 22, 2021
    - [IDETC-CIE 2021](https://event.asme.org/IDETC-CIE) - February 23, 2021
    - [IMECE](https://event.asme.org/IMECE) - March 09, 2021
    - [CDC](https://2021.ieeecdc.org/) - March 18, 2021
* End of the semester

## Running/Jumping Robot Team - Spring 2021
### Logan Sullivan
* Gather data using current system design combined with both printed rigid and flexible designs. Data will be used to evaluate performance differences.
  * subtask
* Assist in the development of the software related to evaluating RL's usefulness in the mechanical design and control on non-rigid robotic systems.
  * OpenAI code 
  * ROS Code
  * OpenAI simulation results

### Jacob LaBerteaux
* Assist in the sim-to-real aspect of the project.
  * 3D modeling and printing
* Ensure the hardware ready when the software is.
  * Motors, drivers, controller boards, wiring harnesses, 3D printed legs, leg joints, body/payload
  * Testing environment
* ensure the architecture is in place for collecting data on sim-to-real process
  * sensors
  * software

### Y H Dang
* Not sure what Y Dang is curious to work on or where his skill set will fit in. Will be polling this soon.


## Crawfish Peeler Team - Spring 2021
### Brennon Moeller
* Continue to assist in the mechanical design and integration of parts to complete a working Crawfish peeler.
* Assist in the testing of designs to evaluate performance to make design alterations for performance improvements.


> * What is your research plan for the semester?
>     - What do you want to accomplish this semester? 
>     - What's the timeline of events/tasks to get you to that semester goal? Identify monthly or bi-weekly milestones to hold yourself to.
>     - What current roadblocks exist on the way to this goal? 
>     - What knowledge do you need to gain to reach these goals? (be as specific as possible)
>     - What external deadlines/milestones exist this term? (Conference papers, UL paperwork, etc)
> 
> * What is your plan for your flexible-legged undergrad team?
>     - What do you want each of them to accomplish this semester?
>     - What are the monthly or bi-weekly milestones/deliverables for each of them?
> 
> * What is your plan for your Crawfish Peeler team?
>     - What do you want each of them to accomplish this semester?
>     - What are the monthly or bi-weekly milestones/deliverables for each of them?
