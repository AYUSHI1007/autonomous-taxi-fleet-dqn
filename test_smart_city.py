from smart_city_env import SmartCityTaxiEnv
import time

env = SmartCityTaxiEnv(grid_size=6)
state, info = env.reset()

print("Initial City Grid:")
env.render()

# Run 5 random moves so you can see how fuel drops and traffic is rendered
for step in range(5):
    action = env.action_space.sample() # Random action
    state, reward, terminated, truncated, _ = env.step(action)
    print(f"\nStep {step + 1}: Action Taken = {action}, Reward = {reward}")
    env.render()
    time.sleep(0.5)

print("\nEnvironment is working cleanly!")