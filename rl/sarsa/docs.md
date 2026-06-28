# SARSA 
- The SARSA **update** is: 

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t
$$
where $\delta_t$ is the **TD Error**: 

$$
\delta_t = R_{t+1} + \gamma Q(s_{t+1}, a_{t+1})-Q(s_t, a_t)
$$

- SARSA loop: 
For each episode: 
    - Initialize state: $S_t =$ environment.reset()
    - Choose action using current policy: $A_t\sim\pi(\dot\lvert S_t)$
    - Take action: $S_{t+1}, R_{t+1}, \text{done} =$ env.step($A_t$)
    - Choose next action using same policy: $A_{t+1}\sim\pi(\dot\lvert S_{t+1})$
    - Update Q-value: 

    $$
    Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_{t+1}+\gammaQ(s_{t+1}, a_{t+1})-Q(s_t, a_t)]
    $$

    - Move forward : $s_t\leftarrow s_{t+1}$ and $a_t\leftarrow a_{t+1}$
    - Repeat until episode ends.

$\rightarrow$ SARSA learns from the **action it actually takes next**

# Expected SARSA
- Expected SARSA **update** is: 

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha[R_{t+1}+\gamma\sum_a\pi(a\lvert s_{t+1})Q(s_{t+1}, a)-Q(s_t, a_t)]
$$ 

$$
\sum_a\pi(a\lvert s_{t+1})Q(s_{t+1}, a) = \mathbb{E}_{a\sim\pi}[Q(s_{t+1}, a)]
$$

The target become: 

$$
R_{t+1}+\gamma\mathbb{E}_{a\sim\pi}[Q(s_{t+1}, a)]
$$

- Expected SARSA loop: 
```text
For each episode: 
    S = env.reset()
    done = False 
    while not done: 
        A = choose_action_epsilon_greedy(Q, S)
        S_next, R, done = env.step(A)
        if not done: 
            expected_q = 0 
            for each action a: 
                prob = policy_probability_epsilon_greedy(Q, S_next, a)
                expected_q += prob * Q[S_next, a]
            target = R + gamma * expected_q
        else: 
            target = R
```

$\rightarrow$ Expected SARSA learns from the **expected value of all possible next actions**
