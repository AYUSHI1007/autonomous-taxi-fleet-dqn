import gymnasium as gym
import numpy as np
import random
import time

# 1. Create the Taxi environment (using Taxi-v4 for newer Gymnasium)
env = gym.make("Taxi-v4", render_mode="ansi")

# 2. Initialize the Q-Table with zeros
state_count = env.observation_space.n
action_count = env.action_space.n
q_table = np.zeros((state_count, action_count))

# 3. Learning Hyperparameters
learning_rate = 0.7
discount_factor = 0.618
epsilon = 1.0
epsilon_decay = 0.001
min_epsilon = 0.01
episodes = 2000

print("🚕 Training the Taxi agent... Please wait a few seconds!")

# 4. The Training Loop
for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])   # Exploit best known move

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update Q-table with the Bellman equation
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + learning_rate * (reward + discount_factor * next_max - old_value)

        state = next_state

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon -= epsilon_decay

print("✅ Training complete! The Taxi is now an expert driver.\n")

# 5. Watch the Trained Taxi Drive!
test_env = gym.make("Taxi-v4", render_mode="human")
state, info = test_env.reset()
done = False
total_reward = 0
step_count = 0

print("Watching the trained taxi complete a live trip...")
while not done:
    test_env.render()
    time.sleep(0.4)
    
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, info = test_env.step(action)
    total_reward += reward
    step_count += 1
    done = terminated or truncated

test_env.render()
print(f"🎉 Trip completed in {step_count} steps with total reward: {total_reward}!")
test_env.close()