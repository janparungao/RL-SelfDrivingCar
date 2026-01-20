# RL Self-Driving Car (CarRacing)

This project was developed as part of the **Designing with Intelligent Agents** module.  
It explores training a **reinforcement learning (RL) agent** to autonomously drive a car around racing tracks of varying difficulty using the **OpenAI Gymnasium `CarRacing` environment**.

The focus of the project is on **agent design decisions**, **learning behaviour**, and **performance comparison** across different track complexities.

---

## Project Objectives

- Train a reinforcement learning agent to complete a racing track
- Compare agent performance on an **easy** and a **hard** track
- Evaluate learning behaviour using:
  - reward curves
  - qualitative driving performance
  - observed policy behaviour

---

## Environment

- **Environment:** OpenAI Gymnasium – `CarRacing`
- **Observation space:** RGB image frames (pixel-based)
- **Original action space:** Continuous (steering, acceleration, brake)
- **Implemented action space:** Discrete (custom-defined actions)

---

## Agent Design Decisions

Several deliberate design choices were made to shape the agent and its learning approach.

### Discrete Action Space

Although the `CarRacing` environment naturally supports **continuous control**, a **discrete action space** was used instead.

This decision was made because it:
- simplifies the learning problem
- enables the use of value-based reinforcement learning methods
- makes agent behaviour easier to analyse and debug

---

### Reinforcement Learning Algorithm: DQN

A **Deep Q-Network (DQN)** was chosen as the learning algorithm due to:

- its suitability for discrete action spaces
- its relatively simple and interpretable structure
- its effectiveness in learning action–value relationships

DQN also allows for straightforward evaluation through:
- reward curves
- policy behaviour
- qualitative inspection of driving performance

---

### Neural Network Architecture

A **Convolutional Neural Network (CNN)** was used to process pixel-based observations from the environment.

This choice was motivated by:
- the widespread use of CNNs in vision-based RL tasks
- their ability to extract spatial features such as:
  - track edges
  - curves
  - road boundaries

---

## Reward Function Design

The reward function was designed to encourage **stable forward progress** while discouraging unsafe or unproductive behaviour.

### The agent is rewarded for:
- progressing along the track
- remaining within track boundaries

### The agent is penalised for:
- leaving the track
- negative progress (moving backwards)

This reward structure encourages consistent driving behaviour and discourages actions that lead to failure or instability.

---

## Evaluation

Agent performance was evaluated using:
- reward curves during training
- visual inspection of driving behaviour
- comparison of learning stability between:
  - an easier track
  - a more complex track

These evaluations helped assess how track difficulty impacts learning efficiency and policy robustness.

---

## Notes & Limitations

- The environment observations are high-dimensional, making learning computationally expensive
- Discretising the action space limits fine-grained control compared to continuous methods
- The project prioritises **learning analysis and design reasoning** over achieving optimal lap times

---

## Possible Improvements

If revisited, future work could include:
- using continuous control methods (e.g. PPO, SAC)
- improving reward shaping for smoother driving
- adding frame stacking or temporal memory
- evaluating generalisation across unseen tracks

---

## Why This Project

This project demonstrates:
- practical reinforcement learning implementation
- thoughtful agent and reward design
- reasoning about trade-offs in RL systems
- analysis of learning behaviour rather than black-box training
