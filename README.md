----------- RL-SelfDrivingCar -----------

This project was for the Designing with Intelligent Agents module coursework. It explores training a reinforcement learning agent to drive a car around two tracks with varying levels of difficulty using OpenAI's Gymnasium CarRacing environment

----------- Project Objectives -----------
- Train a reinforcement agent to complete a racing track
- Compare performance of the model on an 'easy' and 'hard' track
- Evaluate learning behaviour using reward curves and qualitative performance

----------- Agent Design Decisions ----------- 

Through this coursework, several deliberate design choices to form the basis of my intelligent agent and it's learning approach.

----------------------

Although the CarRacing environment naturally supports a continuous control, a discrete action space was chosen instead as:
- It simplified the learning problem
- Enabled the use of value-based reinforcement learning methods

----------------------

A Deep Q-Network (DQN) approach was chosen for the agent reward system due to:
- Its suitability for a discrete action space
- Its simplicity and interpretibility
DQN provides a clear framework for analysing how the agent learns action–value relationships and allows for straightforward evaluation through reward curves and policy behaviour.

----------------------

A Convolutional Neural Network was used to analyse the pixel-based observations from the CarRacing environment. This choice was backed up by:
- Their widespread use in vision-based reinforcement learning agents
- Their ability to extract visual features such as edges, curves and track boundaries

----------------------

The reward function was designed to allow the agent to learn to progress through the track while staying within the track boundaries

The agent is rewarded for:
- Progressing along the track
- Remaining within track boundaries

The agent is penalised for:
- Leaving the track
- Progressing negatively (backwards) on the track

This reward structure encourages consistent driving behavior and discourages actions that lead to failure
