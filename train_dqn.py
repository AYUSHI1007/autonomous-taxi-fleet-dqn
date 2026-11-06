import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
from smart_city_env import SmartCityTaxiEnv

print("🧠 Starting Deep Q-Network (DQN) Training...")
print("The AI will practice 500 trips. Watch the average rewards increase!\n")

# 1. THE AI BRAIN (Neural Network)
class TaxiBrain(nn.Module):
    def __init__(self, input_dim=8, output_dim=7):
        super(TaxiBrain, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. THE MEMORY BOX (Experience Replay)
class MemoryBox:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def save(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 3. SETUP
env = SmartCityTaxiEnv(grid_size=6)
policy_net = TaxiBrain(input_dim=8, output_dim=7)
target_net = TaxiBrain(input_dim=8, output_dim=7)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
criterion = nn.MSELoss()
memory = MemoryBox(capacity=10000)

episodes = 1200
batch_size = 64
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.996
min_epsilon = 0.02
start_time = time.time()

# 4. THE TRAINING LOOP
for episode in range(1, episodes + 1):
    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Ask Brain or Guess?
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_scores = policy_net(state_tensor)
            action = q_scores.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        memory.save(state, action, reward, next_state, done)
        state = next_state

        # Study from memories
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones).unsqueeze(1)

            current_q = policy_net(states).gather(1, actions)
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + (1 - dones) * gamma * max_next_q

            loss = criterion(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 100 == 0:
        print(f"Episode {episode:3d}/{episodes} | Total Reward: {total_reward:6.1f} | Randomness (Epsilon): {epsilon:.2f}")

elapsed = time.time() - start_time
print(f"\n✅ Training finished in {elapsed:.1f} seconds!")

torch.save(policy_net.state_dict(), "taxi_brain.pth")
print("💾 Saved the trained AI brain as 'taxi_brain.pth'")

# 5. WATCH THE TRAINED BRAIN DRIVE
print("\n" + "="*45)
print("🚀 WATCHING THE AI BRAIN COMPLETE A LIVE TRIP")
print("="*45)

state, info = env.reset()
done = False
step = 0
total_reward = 0

action_names = ["North ⬆️", "South ⬇️", "East ➡️", "West ⬅️", "Pick Up 🧍", "Drop Off 🏁", "Refuel ⛽"]

while not done and step < 25:
    env.render()
    time.sleep(0.5)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax().item()

    print(f"Step {step+1}: AI chose -> {action_names[action]}")
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    step += 1
    done = terminated or truncated

env.render()
print(f"\nTrip Finished in {step} steps with Total Reward: {total_reward:.1f}!")