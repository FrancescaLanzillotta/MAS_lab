# Stochastic Value Iteration in a Grid

## Introduction
The goal of this project is to implement the value iteration algorithm in a 6x6 grid, populated by a goal, 
an agent and a number of enemies. Through the algorithm we can define an optimal policy so that our agent
can reach the goal while avoiding the enemies. 

To achieve this in VI, the Bellman equation is used to guide the process of iteratively updating value 
estimates for each state, providing a recursive relationship that expresses the value of a state in 
terms of the values of its neighbouring states.

The main idea behind this algorithm hinges on a modified version of the Bellman optimality equation
 where the value function is iteratively improved by optimizing with respect to the previous decisions.
  
![d](https://latex.codecogs.com/svg.image?&space;V_{k&plus;1}(x)=\max_u\left[r(x,u)&plus;\sum_{x'\in\mathbb{X}}\phi(x'|x,u)V_k(x')\right])
---
## Implementation
In the following image we can see an example of the populated grid where:
- &#9873; is the goal
- ● is the agent
- &#11043; are the enemies
- ✘ is the obstacle

<img src="/img/Base_grid.png" width="500" height="375">

### Synchronous Value Iteration
<img src="/img/sync_plot.png" width="700" height="420">

### Asynchronous Value Iteration
<img src="/img/async_plot.png" width="700" height="420">
