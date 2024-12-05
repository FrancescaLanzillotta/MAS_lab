# Stochastic Value Iteration in a Grid

---

## Introduction
The goal of this project is to implement the value iteration algorithm in a grid, populated by a goal, 
an agent and a fail cell. Both the goal and the fail cell can move randomly within the grid. Through the algorithm we 
can define an optimal policy so that our agent can reach the goal while avoiding failure. 

The main step in algorithm hinges on a modified version of the Bellman optimality equation where the value function is 
iteratively improved by optimizing with respect to the previous decisions.

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large\dpi{600}\bg{white}V_{k&plus;1}(x_a,x_g,x_f)=\max_u\left[r(x_a)&plus;\sum_{x_a',x_g',x_f'\in\,\mathbb{X}}\phi(x_a'|x_a,u)\;\phi(x_g'|x_g)\;\phi(x_f'|x_f)V_k(x_a',x_g',x_f')\right]" width="1000" alt="Value Iteration step formula">
</p>
where

- 𝕏 is the state space.
- 𝕌 is the action space.
- φ is the probability density function expressing the probability of transitioning from one state to another.
In the agent's case, it depends both on agent's position and the action taken. For the goal and the fail state, 
the transition probability depends only on their current position.
- 𝑟 is the immediate reward received by the agent depending on the cell its occupying.


From the optimal values 𝑉 computed in the algorithm, we can extract the optimal policy as 

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large\dpi{600}\bg{white}\gamma(x_a,x_g,x_f)=\text{arg}\max_u\left[r(x_a)&plus;\sum_{x_a',x_g',x_f'\in\,\mathbb{X}}\phi(x_a'|x_a,u)\;\phi(x_g'|x_g)\;\phi(x_f'|x_f)V_k(x_a',x_g',x_f')\right]" width="10000" alt="Optimal Policy formula">
</p>
This means that, for every possible position of the goal and the fail cell on the grid, the agent, depending on its 
position, will always know which action to perform in order to reach the goal while maximizing the expected return.  

--- 
## Implementation
In the following image we can see an example of a 5x5 populated grid where:
- &#9873; is the goal
- <span STYLE="font-size:14.0pt"> ● </span>is the agent
- &#11043; is the fail state

<p align="center">
<img src="/img/Basic-VI-5x5%20grid.png" width="500"  alt="Base grid with a goal, a fail state and an agent">
</p>
The reward for reaching the goal is +1 while the reward for stepping on the fail state is -1. For all the
other cells the reward is -0.04 to incentivize the agent to reach the goal as quickly as possible.

All the experiments were performed on a 5x5 grid with stopping condition ε = 10⁻⁶.

### Synchronous Value Iteration
In the synchronous version of Value Iteration, all states are updated at each iteration. In this 
implementation, the algorithm stops when the change in the values between iterations is less 
than the threshold ε.
The following image shows the optimal values and the optimal policy found for a specific configuration of the goal and 
the fail cell.

<p align="center">
<img src="/img/sync-VI-59%20it.png" width="700" >
</p>

### Asynchronous Value Iteration
In this version, at each iteration one state is picked randomly and only its value is updated. Despite 
increasing the number of iterations for convergence, this algorithm is faster than its
synchronous counterpart. Termination is guaranteed provided that all the states are selected infinitely
 often.

The following image shows the optimal values and the optimal policy found for a specific configuration of the goal and 
the fail cell.

<p align="center">
<img src="/img/async-VI-499375%20it.png" width="700">
</p>

## Reaching the goal
<p align="center">
<img src="/img/async-VI-5x5%20grid.gif" width="500" alt="Agent moving through the grid following the computed policy">
</p>

<p>
This animation shows how the results of the algorithm can be used to move the agent to the goal while avoiding the fail 
cell. All the objects in the grid start from a standard position. The agent, given its position and the positions of the
other objects knows which action to perform. At each step, the goal and the fail cell can move randomly in one of the 
adjacent cells, and we can see how the optimal policy changes accordingly. </p>

---
