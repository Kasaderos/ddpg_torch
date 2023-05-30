from ddpg import DDPG

# Create the DDPG agent
state_dim = 4  # Dimension of the state space
action_dim = 2  # Dimension of the action space
max_action = 1.0  # Maximum action value
agent = DDPG(state_dim, action_dim, max_action)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        agent.train(batch_size)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    print("Episode: {}, Reward: {}".format(episode, episode_reward))

# Save the trained agent
agent.save("trained_agent")

# Load the trained agent
agent.load("trained_agent")

# Evaluation loop
for _ in range(num_eval_episodes):
    state = env.reset()
    episode_reward = 0
    
    for _ in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    print("Episode Reward: {}".format(episode_reward))
