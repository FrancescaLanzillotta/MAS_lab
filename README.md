# Stochastic Value Iteration in a Grid

## Introduction
The goal of this project is to implement the value iteration algorithm in a 6x6 grid, populated by a goal, 
an agent and a number of enemies. Through the algorithm we can define an optimal policy so that our agent
can reach the goal while avoiding obstacles and enemies. 

The main step in algorithm hinges on a modified version of the Bellman optimality equation
 where the value function is iteratively improved by optimizing with respect to the previous decisions.
  
<center><img src="https://latex.codecogs.com/png.image?\dpi{600}\bg{white}&space;V_{k&plus;1}(x)=\max_u\left[r(x,u)&plus;\sum_{x%27\in\mathbb{X}}\phi(x%27|x,u)V_k(x%27)\right]" width="522" height="80">
---
## Implementation
In the following image we can see an example of the populated grid where:
- &#9873; is the goal
- ● is the agent
- &#11043; are the enemies
- ✘ is the obstacle

<center><img src="/img/Base_grid.png" width="500" height="375">

### Synchronous Value Iteration
<center><img src="/img/sync_plot.png" width="700" height="420">

### Asynchronous Value Iteration
<center><img src="/img/async_plot.png" width="700" height="420">
