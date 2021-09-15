## Authors: Andrew Albright, DV
## Date: 04/19/2021
## Due Date: 04/29/2021

# Mechanical Evolution of Flexible-Legged Jumping Systems with Reinforcement Learning

## Abstract: 
Legged systems have many advantages when compared to their wheeled counterparts. For example, they can more easily navigate extreme, uneven terrain. However, there are disadvantages as well, including dramatically lower energy efficiency. To mitigate this performance issue, research has shown that using flexible components not only increases efficiency but also performance. Flexible systems are highly nonlinear however, and are difficult to develop controllers for using traditional methods. Because of the difficulties encountered in modeling flexible systems, control methods such as reinforcement learning can be used which use neural networks to represent the nonlinear model and controller of the systems. Instead of tasking a reinforcement learning algorithm with learning or controlling the system, it can instead be tasked with learning mechanical parameters of a system to match a control input. 

## Introduction:
The use of flexible components within legged locomotive systems has proved useful for both reducing power consumption and increasing performance \cite{Sugiyama2004, Buondonno2017, Hurst2008}. However, designing controller for these systems is difficult as the flexibility of the system generates highly nonlinear models. As such, employing series-elastic-actuators (SEA) instead of flexible links is an attractive and popular solution where the models of the systems become more manageable \cite{Buondonno2017, Zhang2019, Pratt1995}. Still, the use of SEAs do not represent the full capability of flexible systems, and as such other methods are used which use flexible tendon-like materials meant to emulate more organic designs \cite{Iida2005}. Even these however are not representative of fully flexibly links like the ones used in \cite{Saranli2001}.

Control methods have been developed which are designed and work well for flexible systems like the ones mentioned \cite{Luo1993, Modeling2003}. Still, these are challenging to develop as the systems become more complex. As such, work has been done which uses neural networks and methods such as reinforcement learning to develop controllers for flexible systems \cite{Bhagat2019e, Thuruthelb}. Specifically, \cite{Dwiel2019d} showed the use of RL for training higher performing control strategies for flexible legged locomotive systems compared to rigid legged systems. 

In addition to the work done using RL to develop controllers for flexible systems. Work has been completed which shows that techniques such as RL can be used to concurrently design the mechanical aspects of a system and a controller to match said system \cite{Ha2019j}. These techniques have even been used to define mechanical parameters and control strategies where the resulting controller and hardware were deployed in a sim-to-real process, validating the usability of the technique \cite{Chen2020}. Using this technique for legged-locomotion has also been studied, but is limited to the case of rigid systems \cite{Schaff2019e}. 

As such, this paper starts the discovery of using RL for concurrent design of flexible-legged locomotive systems. A simplified flexible jumping system is used where, for the initial work, the control input is held fixed so that the RL algorithm is tasked with only learning optimized mechanical parameters. The rest of the paper is broken down such that in the next section, similar work will be discussed. In Section 3, a brief explanation of the RL algorithm used will be presented. Following that, in Sections 4 and 5, the environment structure will be explained and the experiments will be broken down. Then, in Section 6, results will will be displayed. Finally, in Section 7, a conclusion will be made with implications based on the results.

## Related Work:
Basically a slightly more in depth conversation of each paragraph in the introduction
### Flexible Locomotive Systems
The use of flexible components within robotics systems has shown improvements in performance measures, specifically ones were power consumption are important \cite{Hurst2008}. Specifically, flexible systems have shown advantages in locomotion applications where crawling and jumping are employed \cite{Sugiyama2004}. Previous work has show that the use of flexible components within the legs of legged locomotion systems increase performance while decreasing power consumption \cite{Saranli2001}. Contrasting the use of flexible links within legged systems, much work has been done showing the uses of series-elastic-actuators for locomotive systems \cite{Pratt1995}. Much of this work involving human interaction with robotic systems where rigidity is not always ideal \cite{Zhang2019}. The studies of flexible systems are challenging however, as the models which represent them are often highly nonlinear and therefore difficult to develop control systems for. As such, there is a need for solutions which can be deployed for developing controllers for these nonlinear systems.

### RL Control of Flexible Legged Systems
Control methods developed for flexible systems have been shown to be effective for certain tasks \cite{Luo1993, Ahmadi1997}. However, because of system model nonlinearities, using traditional methods to control flexible systems is not always advantageous. As such, research has been completed that shows the viability of using reinforcement learning to develop the control strategies for flexible systems \cite{Bhagat2019e}. As is with most reinforcement learning applications, different techniques are applied depending on the application. The technique has been used in simple planar cases where it is compared to a PD control strategy for vibration suppression and proves to be a higher performing method \cite{He2020f}. Additionally it has also been shown to be effective at defining control methods for flexible legged locomotion where \cite{Dwiel2019d} used Deep Deterministic Policy Gradient \cite{Lillicrap2016h} to train running strategies for a flexible legged quadruped. Much of the research is based in simulation however, and often the controllers are not deployed in a sim-to-real fashion which leads to the question on weather or not these are practically useful techniques. 

### Concurrent Design of Systems
Defining an optimal controller for a system can be challenging due to things like mechanical and electrical design limits. This is especially true when the system is flexible and the model is nonlinear. A solution to this challenge is to concurrently design a system with the controllers so that the two are jointly optimized and designed for each other. Recent work has been completed which used advanced methods such as evolutionary strategies to define robot design parameters \cite{Wang2019}. Additionally, reinforcement learning has been shown to be a viable solution for concurrent design of locomotive systems in 2D simulation locomotive \cite{Ha2019j}. This is further proved to be a viable method by demonstrating more complex morphology modifications in 3D reaching and locomotive tasks \cite{Schaff2019e}. However these techniques have not been applied to flexible type systems for locomotive tasks. 

## Reinforcement Learning
Brief discussion of the RL technique used.

## Pogo-stick Model/Environment
The pogo-stick model show in Figure X has been shown to be useful as a representation of several different running and jumping gaits \cite{Blickhan1993a}. As such, it is used in this work to demonstrate the ability of reinforcement learning for the initial steps of concurrent design. The models parameters are summarized in Table X. 

The variable m_a represents the mass of the actuator, which moves along the rod with mass m_l. A non-linear spring with constant k is used as the representation of flexibility. A damper (not shown in Figure X) is parallel to the spring. Variable x and x_a represnt the systems vertical position with respect to the ground and the actuators position along the rod, respectively. The system is additionally constrained such that it only moves vertically so the reinforcement agent is not required to balance the system.

The equation of motion describing the system are:

where x and x_ are position and velocity of the rod
respectively, the acceleration of the actuator, xa, is the
control input, and mt is the mass of the complete system.
Ground contact determines the value of , so that the
spring and damper do not supply force while the leg is
airborne:

## Experiments
How the agents were trained
How the agents learn spring K.

## Results
* Spring K the agent learns.
* How the system performs on K values around that spring K.
    * Height reached
    * Power Used ?

## Conclusions

## List of Figures
* Spring K during evaluation
  * Spring K vs Simulation ???
* Performance of K values around the "best" K value
* Time Series Control Input
* Environment Figure
* Jump Type Figures

* We might be able to train agent that are more efficient too?
  * Distribution of power used around "best" K value

## Questions to ask/answer
* How is this to be trained and evaluated?
* How many itterations would be acceptable for us when trying to find an optimal K?
  * 10? 30? 60?
  * Each simulation is roughly ~X seconds, so maybe we base the steps on a time rather than a number of simulations?