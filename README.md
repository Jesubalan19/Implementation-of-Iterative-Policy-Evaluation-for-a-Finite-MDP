# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    
    V = np.zeros(env.observation_space.n)
    iteration = 0

    while True:
        delta = 0
        new_V = np.copy(V)

        for s in range(env.observation_space.n):
            v = 0

            for a, action_prob in enumerate(policy[s]):

                for prob, next_state, reward, terminated in env.P[s][a]:
                    v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not terminated)
                    )

            new_V[s] = v
            delta = max(delta, abs(V[s] - new_V[s]))

        V = new_V
        iteration += 1
        if delta < theta:
            break

    return V, iteration

```

---

## Output

```
Name: Jesubalan A
Register Number: 212223240060
Number of iterations: 71

State-Value Function:
[0.0123561  0.01042443 0.01933841 0.00947773 0.01478703 0.
 0.03889444 0.         0.03260246 0.08433763 0.13781085 0.
 0.         0.17034482 0.43357944 0.        ]
Name: Jesubalan A
Register Number: 212223240060

State-Value Function as 4x4 Grid:
[[0.0124 0.0104 0.0193 0.0095]
 [0.0148 0.     0.0389 0.    ]
 [0.0326 0.0843 0.1378 0.    ]
 [0.     0.1703 0.4336 0.    ]]


```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

Based on the output you obtained, here is a suitable **Inference** for your experiment report:

### **Inference**

The Policy Evaluation algorithm was successfully executed on the 4×4 FrozenLake environment and converged after **71 iterations**. The computed state-value function indicates the expected long-term reward for each state when following the given policy. States closer to the goal have higher state values, with the highest value of **0.4336** observed at state 14, which is adjacent to the goal. States corresponding to holes and terminal states have a value of **0**, as they do not provide any future reward. The gradual increase in state values from the starting state (**0.0124**) toward the goal demonstrates that the Bellman Expectation Equation correctly propagates future rewards backward through the environment. Thus, the experiment verifies that Policy Evaluation accurately estimates the value of each state under a fixed policy and serves as a fundamental step in Dynamic Programming and Reinforcement Learning.

---


