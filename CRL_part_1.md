# introduction to CRL
## part1
  - RL -> CRL
![This is an image](./screenshot1.png)
  - (structural causal model)   SCM 
  ![This is an image](./scs2.png)
  - inverse probability weighting (IPW)

# Causal inference
- if Y is a child of X, then X is the direct cause of Y
- if Y is the descendent of X, the X is a potential cause of Y

# Sarsa(on-policy)
- Derive TD target
  - $Q_\pi(s_t,a_t) = \mathbb E{[R_t+\gamma\cdot Q_\pi(S_{t+1},A_{t+1})]} \approx y_t$, for all $\pi$
  - $\approx r_t+\gamma\cdot Q_\pi(s_{t+1},a_{t+1})$
  - encourage $Q_\pi(s_t,a_t)$ to approximate $y_t$
- Tabular version
  - observe a transition $(s_t,a_t,r_t,s_{t+1})$
  - sample $a_{t+1} \sim \pi(\cdot|s_{t+1})$, where $\pi$ is the policy function
  - TD target: $y_t=r_t+\gamma\cdot Q_\pi(s_{t+1},a_{t+1})$
  - TD error: $\delta_t = Q_\pi(s_t,a_t )-y_t$
  - Update: $Q_\pi(s_t,a_t) \leftarrow Q_\pi(s_t,a_t)-\alpha\cdot \delta_t$
- Value network version
  - Approximate $Q_\pi(s_t,a_t)$ by the value network, $q(s,a;\omega)$
  - learn $\omega$
- TD error and Gradient
  - TD target: $y_t = r_t +\gamma\cdot q(s_{t+1},a_{t+1};\omega)$
  - TD error: $\delta_t = q(s_t,a_t;\omega)-y_t$
  - Loss: $\delta^2_t/2$
  - Gradient: $\frac{\partial \delta^2_t/2}{\partial \omega}=\delta_t\cdot \frac{\partial q(s_t,a_t;\omega)}{\partial \omega}$
  - Gradient descent: $\omega \leftarrow \omega-\alpha\cdot \delta_t\cdot \frac{\partial q(s_t,a_t;\omega)}{\partial \omega}$
# Q-learning(off-policy)
- Sarsa VS Q-learning
  - Sarsa is for training action-value function, $Q_\pi(s,a)$
  - TD target: $y_t = r_t +\gamma\cdot q(s_{t+1},a_{t+1};\omega)$
  - Used Sarsa for updating value network(critic)
  - Q-learning is for training the optimal action-value function, $Q^\ast(s,a)$
  - TD target: $y_t = r_t +\gamma\cdot \max\limits_a Q^\ast(s_{t+1},a)$
  - Used Q-learning for updating DQN
- Derive TD target
  - $Q_\pi(s_t,a_t)= \mathbb E[R_t+\gamma\cdot Q_\pi(S_{t+1},A_{t+1})]$ for all $\pi$
  - if $\pi$ is the optimal policy $\pi^\ast$, then <br> &emsp;$Q_{\pi^\ast}(s_t,a_t)= \mathbb E[R_t+\gamma\cdot Q_{\pi^\ast}(S_{t+1},A_{t+1})]$ 
  - $Q_{\pi^\ast}$ and $Q^\ast$ both denote the optimal action-value function
  - identity: $Q^\ast(s_t,a_t)= \mathbb E[R_t+\gamma\cdot Q^\ast(S_{t+1},A_{t+1})]$ 
    - The action $A_{t+1} = \argmax\limits_aQ^\ast(S_{t+1},a)$
    - $Q^\ast(s_t,a_t)= \mathbb E[R_t+\gamma\cdot \max\limits_aQ^\ast(S_{t+1},a)]$
    - $\approx r_t+\gamma\cdot \max\limits_aQ^\ast(s_{t+1},a)$ which is the TD target $y_t$
  - Tabular version
    - observe a transition $(s_t,a_t,r_t,s_{t+1})$
    - TD target: $y_t =r_t+\gamma\cdot \max\limits_aQ^\ast(s_{t+1},a)$
    - TD error: $\delta_t = Q^\ast(s_t,a_t)-y_t$
    - Update: $Q^\ast(s_t,a_t)\leftarrow Q^\ast(s_t,a_t)-\alpha\cdot \delta_t$
  - DQN version
    - Approximate $Q^\ast(s_t,a_t)$ by DQN, $Q(s,a;\omega)$
    - DQN controls the agent by:$a_t=\argmax\limits_aQ(s_t,a;\omega)$
    - learn $\omega$
    - Observe a transition $(s_t,a_t,r_t,s_{t+1})$
    - TD target: $y_t =r_t+\gamma\cdot \max\limits_aQ(s_{t+1},a;\omega)$
    - TD error: $\delta_t = Q(s_t,a_t;\omega)-y_t$
    - Update: $\omega\leftarrow \omega-\alpha\cdot \delta_t\cdot \frac{\partial Q(s_t,a_t;\omega)}{\partial \omega}$
# Multi-step
- Multi-step return
  - $U_t = R_t + \gamma\cdot U_{t+1}$
  - $U_t = \sum^{m-1}_{i=0}\gamma^i\cdot R_{t+i}+\gamma^m\cdot U_{t+m}$
- Multi-step TD targets
  - m-step TD target for Sarsa:<br>&emsp;$y_t = \sum^{m-1}_{i=0}\gamma^i\cdot r_{t+i}+\gamma^m\cdot Q_\pi(s_{t+m},a_{t+m})$
  - m_step TD target for Q-learning:<br>&emsp;$y_t = \sum^{m-1}_{i=0}\gamma^i\cdot r_{t+i}+\gamma^m\cdot \max\limits_a Q^\ast(s_{t+m},a)$
# Actor-critic
## value-based methods(critic)
  - Use neural net $q(s,a;\omega)$ to approximate $Q_{\pi}(s,a)$
  - $\omega$ is the trainable parameters of the neural net
  - input: state s and action a
  - approximate action-value (scalar)
## Policy-based methods(actor)
  - Use neural net $\pi(a|s;\theta)$ to approximate $\pi(a|s)$
  - $\theta$ is the trainable parameters of the neural net
  - input: state s
  - output: probability distributions over the actions
## Network training
- observe state $s_t$
- Randomly sample action $a_t$ according to $\pi(\cdot|s_t;\theta_t)$
- perform $a_t$ and observe new state $s_{t+1}$ and reward $r_t$
- update value network q use TD
  - compute $q(s_t,a_t;\omega_t)$ and $q(s_{t+1},a_{t+1};\omega_t)$
  - TD target: $y_t = r_t +\gamma\cdot q(s_{t+1},a_{t+1};\omega_t)$
  - Loss: $L(\omega)=\frac{1}{2}[q(s_t,a_t;\omega)-y_t]^2$
  - Gradient descent: $\omega_{t+1} = \omega_t-\alpha\cdot \frac{\partial L(\omega)}{\partial \omega}\Bigr|_{\substack{\omega=\omega_t}}$
- update policy network $\pi$ using policy gradient
  - Definition: State-value function approximated using neural networks
    - $V(s;\theta,\omega)=\sum_a\pi(a|s;\theta)\cdot q(s,a;\omega)$
  - policy gradient: derivative of $V(s_t;\theta,\omega)$ w.r.t. $\theta$. 
    - let $g(a,\theta) = \frac{\partial log~{\pi(a|s,\theta)}}{\partial \theta}\cdot q(s_t,a;\omega)$
    - $\frac{\partial V(s_t;\theta,\omega_t)}{\partial \theta} = \mathbb{E}_A{[g(A,\theta)]}$
  - random sampling: $a \sim \pi(\cdot|s_t;\theta_t)$
  - stochastic gradient ascent: $\theta_{t+1}=\theta_t+\beta\cdot g(a,\theta_t)$
- summary of Algorithm
  1. Observe state $s_t$ and randomly sample $a_t\sim \pi(\cdot|s_t;\theta_t)$.
  2.  Perform $a_t$ and observe new state $s_{t+1}$ and reward $r_t$
  3. Randomly sample $\tilde{a}_{t+1} \sim \pi(\cdot|s_{t+1};\theta_t)$
  4.  Evaluate value network: $q(s_t,a_t;\omega_t)$ and $q(s_{t+1},a_{t+1};\omega_t)$
  5.  Compute TD error: $\delta_t = q_t - (r_t+\gamma\cdot q_{t+1})$
  6.  Differentiate value network: $d_{w,t}=\frac{\partial q(s_t,a_t;\omega)}{\partial \omega}\bigr|_{\substack{\omega=\omega_t}}$
  7.  Update value network: $\omega_{t+1} = \omega_t -\alpha\cdot \delta_t \cdot d_{w,t}$
  8.  Differentiate policy network: $d_{\theta,t}= \frac{\partial log~{\pi(a_t|s_t,\theta)}}{\partial \theta} \bigr|_{\substack \theta = \theta_t}$
  9.  Update policy network: $\theta_{t+1}=\theta_t+\beta\cdot \delta_t \cdot d_{\theta,t}$
# Paper
confounded components (c-component), assign two variables to the same group iff they are connected by a path composed solely of bi-directional arrows.