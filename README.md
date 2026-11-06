# Autonomous Taxi Fleet: Route Optimization with Deep RL

Most reinforcement learning tutorials for the classic "Taxi" problem use OpenAI Gym's basic 5x5 grid. It's a great "hello world", but it doesn't reflect how real ride-sharing systems (like Uber, Lyft, or Waymo) actually work:
* In the real world, roads have traffic jams.
* Cars have fuel limits and need to recharge.
* Customers get impatient if they wait too long.
* And there are multiple cars sharing the same streets.

I built this project to take the classic Taxi problem from a toy tabular example to a full multi-agent simulation running on PyTorch Deep Q-Networks, complete with an interactive Pygame telemetry dashboard and real Manhattan street routing via OpenStreetMap.

---

## 🛠️ The Core Progression

### 1. The Baseline (`taxi_rl.py`)
Started with standard Q-learning on Gymnasium's `Taxi-v4`. A lookup table works fine here because a 5x5 grid with fixed pickup points only has 500 possible states. It learned optimal pickup-dropoff paths in ~12 steps.

### 2. Building a Living City (`smart_city_env.py`)
To make things realistic, I built a custom environment that added:
* **Dynamic Traffic Congestion**: Streets randomly shift between clear (1x fuel), moderate (2x fuel), and heavy jams (4x fuel).
* **Fuel & Charging**: Taxis start with a 40-unit tank. Running empty terminates the ride with a heavy penalty. A charging station at `(0, 0)` allows pit stops.
* **Passenger Wait Penalties**: Every second a passenger waits on the curb deducts points, forcing the agent to prioritize urgent pickups.

### 3. Scaling to Deep Q-Networks (`train_dqn.py`)
Adding coordinates, fuel levels, and dynamic traffic blew the state space up from 500 to **over 400,000 combinations**—making a spreadsheet Q-table impossible to train.
* Built a 3-layer neural network in PyTorch (`TaxiBrain`).
* Implemented an Experience Replay buffer (10,000 capacity) to break correlation between consecutive moves.
* **The "Gas Station Loitering" Bug**: Early in training, the agent discovered a funny local minimum—it realized that driving into traffic risked running out of gas (-40 penalty), but parking at the gas pump and refueling continuously only cost a tiny penalty. It literally parked at the pump and refused to leave!
* **The Fix**: Added potential-based reward shaping (Manhattan-distance breadcrumbs toward the target) and penalized refueling on a full tank. The agent immediately climbed to a consistent +137 average reward.

### 4. Interactive Desktop Dashboard (`visualize_taxi.py`)
Built a Pygame interface to watch the trained agent drive in real-time with:
* Live fuel bar and customer wait timer.
* Traffic congestion indicators.
* Interactive controls: press `SPACE` to generate random trips, and `UP`/`DOWN` to change playback speed.

### 5. Multi-Agent Fleet Dispatch (`multi_taxi_sim.py`)
Scaled the system from 1 car to a decentralized 2-car fleet:
* **Shared Brain Policy**: Both vehicles use the same trained DQN weights independently.
* **Dynamic Task Allocation**: Cabs evaluate distance to waiting calls and claim the nearest unassigned passenger without duplicate dispatches.
* **Deadlock Breaker**: Solved a classic head-on standoff where two cars met in a narrow corridor and both yielded forever by adding lateral evasion logic.
* **Auto Pit-Stops**: Cabs with low fuel break off to recharge and release customer reservations to healthy cars.

### 6. Real-World Manhattan Routing (`real_map_taxi.py`)
Ingested real OpenStreetMap street networks around Times Square / Broadway using OSMnx and NetworkX:
* Streets are weighted by real-time traffic speeds (5 km/h for jams vs 40 km/h for clear avenues).
* Added an animated video-game style follow-cam that tracks the taxi as it drives down actual Manhattan streets and turns corners.

---

## 💻 Tech Stack

* **Language**: Python 3.12
* **Reinforcement Learning**: Gymnasium, Custom `gym.Env`
* **Deep Learning**: PyTorch (`torch.nn`, Adam optimizer)
* **GUI & Animation**: Pygame, Matplotlib
* **Spatial & Graph Routing**: OSMnx, NetworkX

---

## 🚦 How to Run It

### Setup
Clone the repo and install the dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/autonomous-taxi-fleet-dqn.git
cd autonomous-taxi-fleet-dqn
pip install gymnasium pygame torch numpy matplotlib osmnx networkx
```

### 1. Train the DQN Brain
Trains the 3-layer neural network across 1,200 episodes (~50 seconds on CPU):
```bash
python train_dqn.py
```
*(Saves the trained model to `taxi_brain.pth`)*

### 2. Run the Multi-Taxi Fleet Simulator
Watch two cabs coordinate, manage fuel, and serve continuous passenger waves:
```bash
python multi_taxi_sim.py
```
* `SPACE`: Reset / spawn new wave
* `UP` / `DOWN`: Adjust car speed
* `P`: Pause / Resume

### 3. Run Real Manhattan Street Navigation
Watch the taxi navigate real-world Times Square road networks with live follow-cam:
```bash
python real_map_taxi.py
```

---

## 💡 Lessons Learned & Engineering Takeaways

1. **Reward design is tricky**: If your penalty for dying is too harsh compared to idling, an RL agent will often find creative ways to "do nothing safely" (like our gas pump loitering bug). Reward shaping with distance heuristics was essential.
2. **Multi-agent edge cases**: In single-agent setups you only care about walls. In multi-agent setups, you quickly run into reservation locks and spatial deadlocks that require clean fallback rules.
3. **From toy grids to real graphs**: Moving from a discrete 2D grid to a graph of GPS coordinates requires a shift in how you think about actions—from simple directions (North/South) to graph-edge traversal based on time weights.
