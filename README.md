# Stochastic Value Iteration in a Grid

---

## Introduction
The goal of this project is to implement the value iteration algorithm in a 6x6 grid, populated by a goal, 
an agent and a number of enemies. Through the algorithm we can define an optimal policy so that our agent
can reach the goal while avoiding obstacles and enemies. 

The main step in algorithm hinges on a modified version of the Bellman optimality equation
 where the value function is iteratively improved by optimizing with respect to the previous decisions.

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large\dpi{600}\bg{white}V_{k&plus;1}(x_a,x_g,x_f)=\max_u\left[r(x_a,u)&plus;\sum_{x_a',x_g',x_f'\in\,\mathbb{X}}\phi(x_a'|x_a,u)\;\phi(x_g'|x_g)\;\phi(x_f'|x_f)V_k(x_a',x_g',x_f')\right]" width="800" alt="Value Iteration step formula">
</p>
where

- 𝑥, 𝑥' ∈ 𝕏 (state space)
- 𝑢 ∈ 𝕌 (action space)
- φ(𝑥'|𝑥,𝑢) is the probability density function expressing the probability of transitioning from state 𝑥 to state 
𝑥' when 𝑢 is applied
- 𝑟(𝑥,𝑢) is the immediate reward received by the agent in state 𝑥 when taking action 𝑢

From the optimal values 𝑉 computed in the algorithm, we can extract the optimal policy as 

<p align="center">
<img src="https://latex.codecogs.com/png.image?\large\dpi{300}\bg{white}\gamma(x_a,x_g,x_f)=\text{arg}\max_u\left[r(x_a,u)&plus;\sum_{x_a',x_g',x_f'\in\,\mathbb{X}}\phi(x_a'|x_a,u)\;\phi(x_g'|x_g)\;\phi(x_f'|x_f)V_k(x_a',x_g',x_f')\right]V_{k&plus;1}(x_a,x_g,x_f)=\max_u\left[r(x_a,u)&plus;\sum_{x_a',x_g',x_f'\in\,\mathbb{X}}\phi(x_a'|x_a,u)\;\phi(x_g'|x_g)\;\phi(x_f'|x_f)V_k(x_a',x_g',x_f')\right]" width="10000" alt="Optimal Policy formula">
</p>


--- 
## Implementation
In the following image we can see an example of the populated grid where:
- &#9873; is the goal
- <span STYLE="font-size:14.0pt"> ● </span>is the agent
- &#11043; are the enemies
- ✘ is the obstacle

<p align="center">
<img src="/img/Base_grid.png" width="500" height="375">
</p>
The reward for reaching the goal is +1 while the reward for being captured by the enemy is -1. For all the
other cells, the reward is -0.04 to incentivize the agent to quickly reach the goal.

### Synchronous Value Iteration
In the synchronous version of Value Iteration, all states are updated at each iteration. In this 
implementation, the algorithm stops when the change in the values between iterations is less 
than the threshold ε.

<p align="center">
<img src="/img/sync_plot.png" width="700" height="420">
</p>

### Asynchronous Value Iteration
In this version, at each iteration one state is picked randomly and only its value is updated. Despite 
increasing the number of iterations for convergence, this algorithm is faster than its
synchronous counterpart. Termination is guaranteed provided that all the states are selected infinitely
 often.

<p align="center">
<img src="/img/async_plot.png" width="700" height="420">
</p>

---