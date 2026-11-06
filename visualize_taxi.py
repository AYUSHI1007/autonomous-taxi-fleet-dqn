import pygame
import torch
import torch.nn as nn
import numpy as np
import time
import sys
from smart_city_env import SmartCityTaxiEnv

# 1. RECREATE THE AI BRAIN ARCHITECTURE
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

# 2. LOAD YOUR TRAINED BRAIN
brain = TaxiBrain(input_dim=8, output_dim=7)
try:
    brain.load_state_dict(torch.load("taxi_brain.pth"))
    brain.eval()
    print("✅ Loaded trained 'taxi_brain.pth' successfully!")
except Exception as e:
    print(f"❌ Error loading taxi_brain.pth: {e}")
    sys.exit()

# 3. INITIALIZE PYGAME
pygame.init()
GRID_SIZE = 6
CELL_SIZE = 90
MAP_WIDTH = GRID_SIZE * CELL_SIZE
MAP_HEIGHT = GRID_SIZE * CELL_SIZE
SIDEBAR_WIDTH = 260
WINDOW_WIDTH = MAP_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = MAP_HEIGHT

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Smart City Taxi - Deep Reinforcement Learning")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Segoe UI", 18, bold=True)
title_font = pygame.font.SysFont("Segoe UI", 22, bold=True)
huge_font = pygame.font.SysFont("Segoe UI", 36, bold=True)

# COLOR PALETTE
COLOR_BG = (28, 30, 38)
COLOR_ROAD = (45, 52, 64)
COLOR_GRID_LINE = (35, 40, 50)
COLOR_CLEAR = (50, 75, 60)
COLOR_TRAFFIC_MED = (180, 140, 40)
COLOR_TRAFFIC_JAM = (190, 50, 50)
COLOR_SIDEBAR = (20, 22, 28)
COLOR_WHITE = (240, 245, 255)
COLOR_GOLD = (255, 204, 0)
COLOR_GREEN = (46, 204, 113)
COLOR_BLUE = (52, 152, 219)
COLOR_RED = (231, 76, 60)

env = SmartCityTaxiEnv(grid_size=GRID_SIZE)
state, _ = env.reset()

action_labels = ["North ⬆️", "South ⬇️", "East ➡️", "West ⬅️", "Pick Up 🧍", "Drop Off 🏁", "Refuel ⛽"]
last_action_text = "Starting Trip..."
total_reward = 0
step_count = 0
done = False
trip_completed = False
speed_delay = 0.35  # Seconds per move
last_move_time = time.time()
paused = False

def draw_hud():
    # Sidebar Background
    pygame.draw.rect(screen, COLOR_SIDEBAR, (MAP_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
    pygame.draw.line(screen, (60, 65, 80), (MAP_WIDTH, 0), (MAP_WIDTH, WINDOW_HEIGHT), 2)
    
    # Title
    t1 = title_font.render("AI TAXI DISPATCH", True, COLOR_GOLD)
    screen.blit(t1, (MAP_WIDTH + 20, 20))
    
    # Trip Status Badge
    status_text = "PICKING UP 🧍" if not env.passenger_in_taxi else "DELIVERING 🏁"
    if trip_completed:
        status_text = "COMPLETED! 🎉"
    status_color = COLOR_GREEN if env.passenger_in_taxi else COLOR_BLUE
    if trip_completed: status_color = COLOR_GOLD
    
    pygame.draw.rect(screen, status_color, (MAP_WIDTH + 20, 60, 220, 36), border_radius=8)
    st_render = font.render(status_text, True, (0, 0, 0))
    screen.blit(st_render, (MAP_WIDTH + 35, 67))
    
    # Stats
    y = 120
    stats = [
        f"Last Action: {last_action_text}",
        f"Steps Taken: {step_count}",
        f"Total Score: {total_reward:.1f}",
        f"Customer Wait: {env.wait_time}s",
    ]
    for s in stats:
        txt = font.render(s, True, COLOR_WHITE)
        screen.blit(txt, (MAP_WIDTH + 20, y))
        y += 35

    # Fuel Bar
    y += 10
    fuel_label = font.render(f"Fuel Gauge: {env.fuel}/{env.max_fuel}", True, COLOR_WHITE)
    screen.blit(fuel_label, (MAP_WIDTH + 20, y))
    y += 28
    
    bar_width = 210
    bar_height = 20
    fuel_ratio = max(0, env.fuel / env.max_fuel)
    fill_width = int(bar_width * fuel_ratio)
    
    bar_color = COLOR_GREEN if fuel_ratio > 0.4 else (COLOR_TRAFFIC_MED if fuel_ratio > 0.2 else COLOR_RED)
    pygame.draw.rect(screen, (50, 55, 65), (MAP_WIDTH + 20, y, bar_width, bar_height), border_radius=6)
    pygame.draw.rect(screen, bar_color, (MAP_WIDTH + 20, y, fill_width, bar_height), border_radius=6)
    
    # Instructions Footer
    y = WINDOW_HEIGHT - 110
    hints = [
        "Controls:",
        "• SPACE: New Random Trip",
        "• P: Pause / Resume",
        "• UP/DOWN: Speed +/-"
    ]
    for h in hints:
        txt = font.render(h, True, (150, 160, 180))
        screen.blit(txt, (MAP_WIDTH + 20, y))
        y += 24

def draw_city():
    # Draw Streets and Traffic Heatmap
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            traffic = env.traffic_grid[r, c]
            
            tile_color = COLOR_ROAD
            if traffic == 1:
                tile_color = (60, 58, 45) # Subtle yellow
            elif traffic == 2:
                tile_color = (75, 40, 40) # Subtle red
                
            pygame.draw.rect(screen, tile_color, rect)
            pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

            # Draw small traffic dots
            center = rect.center
            if traffic == 1:
                pygame.draw.circle(screen, COLOR_TRAFFIC_MED, (rect.right - 14, rect.top + 14), 5)
            elif traffic == 2:
                pygame.draw.circle(screen, COLOR_TRAFFIC_JAM, (rect.right - 14, rect.top + 14), 7)

    # 1. DRAW CHARGING STATION (0, 0)
    station_rect = pygame.Rect(env.station_pos[1] * CELL_SIZE + 10, env.station_pos[0] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.rect(screen, (40, 116, 166), station_rect, border_radius=10)
    st_text = font.render("⛽ GAS", True, COLOR_WHITE)
    screen.blit(st_text, (station_rect.x + 10, station_rect.y + 22))

    # 2. DRAW DESTINATION
    dest_rect = pygame.Rect(env.destination_pos[1] * CELL_SIZE + 10, env.destination_pos[0] * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
    pygame.draw.rect(screen, (142, 68, 173), dest_rect, border_radius=10)
    dest_text = font.render("🏁 GOAL", True, COLOR_WHITE)
    screen.blit(dest_text, (dest_rect.x + 8, dest_rect.y + 22))

    # 3. DRAW PASSENGER (if not inside taxi)
    if not env.passenger_in_taxi:
        pass_x = env.passenger_pos[1] * CELL_SIZE + CELL_SIZE // 2
        pass_y = env.passenger_pos[0] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, (241, 196, 15), (pass_x, pass_y), 20)
        p_char = font.render("🧍", True, (0, 0, 0))
        screen.blit(p_char, (pass_x - 10, pass_y - 12))

    # 4. DRAW TAXI
    taxi_x = env.taxi_pos[1] * CELL_SIZE + 15
    taxi_y = env.taxi_pos[0] * CELL_SIZE + 15
    taxi_rect = pygame.Rect(taxi_x, taxi_y, CELL_SIZE - 30, CELL_SIZE - 30)
    
    # Taxi body
    pygame.draw.rect(screen, COLOR_GOLD, taxi_rect, border_radius=12)
    pygame.draw.rect(screen, (0, 0, 0), taxi_rect, 2, border_radius=12)
    
    # Taxi Cab Roof Light
    roof_rect = pygame.Rect(taxi_x + 15, taxi_y + 8, 30, 10)
    pygame.draw.rect(screen, (255, 255, 255), roof_rect, border_radius=3)
    
    # Passenger aboard indicator
    if env.passenger_in_taxi:
        in_text = font.render("🧍", True, (0, 0, 0))
        screen.blit(in_text, (taxi_x + 22, taxi_y + 22))
    else:
        taxi_text = font.render("TAXI", True, (0, 0, 0))
        screen.blit(taxi_text, (taxi_x + 12, taxi_y + 24))

# 4. MAIN INTERACTIVE LOOP
running = True
print("\n🎮 Pygame Visualizer launched! Press SPACE to restart a new trip anytime.")

while running:
    current_time = time.time()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                state, _ = env.reset()
                total_reward = 0
                step_count = 0
                done = False
                trip_completed = False
                last_action_text = "New Trip Started!"
            elif event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_UP:
                speed_delay = max(0.05, speed_delay - 0.05)
            elif event.key == pygame.K_DOWN:
                speed_delay = min(1.0, speed_delay + 0.05)

    # Agent Step Logic
    if not done and not paused and (current_time - last_move_time >= speed_delay):
        last_move_time = current_time
        
        # Brain evaluates the city situation
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = brain(state_tensor).argmax().item()
            
        last_action_text = action_names = action_labels[action]
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            done = True
            trip_completed = (reward >= 50)  # High reward means successful delivery!

    # Render Frame
    screen.fill(COLOR_BG)
    draw_city()
    draw_hud()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()