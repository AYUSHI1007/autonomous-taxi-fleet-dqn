import pygame
import torch
import torch.nn as nn
import numpy as np
import random
import time
import sys

# 1. LOAD TRAINED BRAIN
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

brain = TaxiBrain(input_dim=8, output_dim=7)
try:
    brain.load_state_dict(torch.load("taxi_brain.pth"))
    brain.eval()
    print("✅ Loaded trained 'taxi_brain.pth' for the fleet!")
except Exception as e:
    print(f"❌ Error loading taxi_brain.pth: {e}")
    sys.exit()

# 2. SETUP
GRID_SIZE = 6
CELL_SIZE = 95
MAP_WIDTH = GRID_SIZE * CELL_SIZE
MAP_HEIGHT = GRID_SIZE * CELL_SIZE
SIDEBAR_WIDTH = 270
WINDOW_WIDTH = MAP_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = MAP_HEIGHT

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Smart City Multi-Taxi Fleet - Deep RL")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Segoe UI", 16, bold=True)
title_font = pygame.font.SysFont("Segoe UI", 20, bold=True)

# PALETTE
COLOR_BG = (25, 28, 36)
COLOR_ROAD = (42, 48, 60)
COLOR_GRID_LINE = (32, 36, 46)
COLOR_SIDEBAR = (18, 20, 26)
COLOR_WHITE = (240, 245, 255)
COLOR_GOLD = (255, 204, 0)
COLOR_CYAN = (0, 210, 255)
COLOR_GREEN = (46, 204, 113)
COLOR_RED = (231, 76, 60)
COLOR_TRAFFIC_MED = (180, 140, 40)
COLOR_TRAFFIC_JAM = (190, 50, 50)

class TaxiAgent:
    def __init__(self, taxi_id, name, color, start_pos):
        self.id = taxi_id
        self.name = name
        self.color = color
        self.pos = list(start_pos)
        self.fuel = 40
        self.max_fuel = 40
        self.passenger = None
        self.completed_rides = 0
        self.total_score = 0
        self.status = "Searching..."
        self.going_to_refuel = False

class Passenger:
    def __init__(self, p_id, pos, dest, color):
        self.id = p_id
        self.pos = list(pos)
        self.dest = list(dest)
        self.color = color
        self.picked_up = False
        self.delivered = False
        self.assigned_taxi = None

station_pos = [0, 0]
traffic_grid = np.random.choice([0, 1, 2], size=(GRID_SIZE, GRID_SIZE), p=[0.75, 0.20, 0.05])

def random_city_spot():
    while True:
        spot = [random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)]
        if spot != [0, 0]:
            return spot

def spawn_single_passenger(p_id, color):
    pos = random_city_spot()
    dest = random_city_spot()
    while dest == pos:
        dest = random_city_spot()
    return Passenger(p_id, pos, dest, color)

def init_world():
    taxis = [
        TaxiAgent(1, "Cab #1 (Gold)", COLOR_GOLD, [0, 5]),
        TaxiAgent(2, "Cab #2 (Cyan)", COLOR_CYAN, [5, 0])
    ]
    passengers = [
        spawn_single_passenger("A", (241, 196, 15)),
        spawn_single_passenger("B", (230, 126, 34))
    ]
    return taxis, passengers

taxis, passengers = init_world()
speed_delay = 0.35
last_move_time = time.time()
paused = False
step_counter = 0

def get_obs(taxi, target_pos, dest_pos):
    p_row = target_pos[0] if not taxi.passenger else -1.0
    p_col = target_pos[1] if not taxi.passenger else -1.0
    
    return np.array([
        taxi.pos[0] / GRID_SIZE,
        taxi.pos[1] / GRID_SIZE,
        p_row / GRID_SIZE if p_row != -1.0 else -1.0,
        p_col / GRID_SIZE if p_col != -1.0 else -1.0,
        dest_pos[0] / GRID_SIZE,
        dest_pos[1] / GRID_SIZE,
        1.0 if taxi.passenger else 0.0,
        max(0.2, taxi.fuel / taxi.max_fuel)
    ], dtype=np.float32)

def move_taxi(taxi, action, other_taxi):
    traffic = traffic_grid[taxi.pos[0], taxi.pos[1]]
    burn = 1 if traffic == 0 else (2 if traffic == 1 else 3)
    
    new_pos = list(taxi.pos)
    if action == 0:   new_pos[0] = max(0, taxi.pos[0] - 1)
    elif action == 1: new_pos[0] = min(GRID_SIZE - 1, taxi.pos[0] + 1)
    elif action == 2: new_pos[1] = min(GRID_SIZE - 1, taxi.pos[1] + 1)
    elif action == 3: new_pos[1] = max(0, taxi.pos[1] - 1)

    # DEADLOCK RESOLUTION: If adjacent car blocks path, sidestep!
    if new_pos == other_taxi.pos:
        taxi.status = "Yielding Road 🛑"
        if taxi.id == 2:
            side_col = taxi.pos[1] + 1 if taxi.pos[1] < GRID_SIZE - 1 else taxi.pos[1] - 1
            sidestep_pos = [taxi.pos[0], side_col]
            if sidestep_pos != other_taxi.pos:
                taxi.pos = sidestep_pos
                taxi.status = "Detouring around T1 🔄"
        return

    taxi.pos = new_pos
    taxi.fuel = max(0, taxi.fuel - burn)
    taxi.total_score -= burn

    # ROADSIDE ASSISTANCE: If fuel hits 0, emergency tow service refuels it!
    if taxi.fuel <= 0:
        taxi.fuel = taxi.max_fuel
        taxi.total_score -= 30  # Tow fee
        taxi.status = "Roadside Tow 🚜"

def draw_hud():
    pygame.draw.rect(screen, COLOR_SIDEBAR, (MAP_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
    pygame.draw.line(screen, (55, 60, 75), (MAP_WIDTH, 0), (MAP_WIDTH, WINDOW_HEIGHT), 2)

    title = title_font.render("FLEET DISPATCH HUD", True, COLOR_WHITE)
    screen.blit(title, (MAP_WIDTH + 20, 15))

    total_completed = sum(t.completed_rides for t in taxis)
    f_stat = font.render(f"Fleet Completed Rides: {total_completed}", True, COLOR_GREEN)
    screen.blit(f_stat, (MAP_WIDTH + 20, 48))

    y = 85
    for taxi in taxis:
        pygame.draw.rect(screen, (30, 34, 44), (MAP_WIDTH + 15, y, SIDEBAR_WIDTH - 30, 140), border_radius=8)
        pygame.draw.circle(screen, taxi.color, (MAP_WIDTH + 30, y + 20), 8)
        
        t_name = font.render(taxi.name, True, taxi.color)
        screen.blit(t_name, (MAP_WIDTH + 45, y + 10))

        status_txt = font.render(f"Status: {taxi.status}", True, COLOR_WHITE)
        screen.blit(status_txt, (MAP_WIDTH + 25, y + 36))

        score_txt = font.render(f"Score: {taxi.total_score:.1f} | Rides: {taxi.completed_rides}", True, (200, 210, 225))
        screen.blit(score_txt, (MAP_WIDTH + 25, y + 62))

        f_ratio = max(0, taxi.fuel / taxi.max_fuel)
        f_color = COLOR_GREEN if f_ratio > 0.4 else (COLOR_TRAFFIC_MED if f_ratio > 0.2 else COLOR_RED)
        pygame.draw.rect(screen, (50, 55, 65), (MAP_WIDTH + 25, y + 92, 200, 16), border_radius=4)
        pygame.draw.rect(screen, f_color, (MAP_WIDTH + 25, y + 92, int(200 * f_ratio), 16), border_radius=4)
        
        f_label = font.render(f"Fuel: {taxi.fuel}/{taxi.max_fuel}", True, COLOR_WHITE)
        screen.blit(f_label, (MAP_WIDTH + 25, y + 112))

        y += 155

    y = WINDOW_HEIGHT - 85
    hints = [
        "Controls:",
        "• SPACE: Reset World",
        "• UP/DOWN: Speed +/-",
        "• P: Pause"
    ]
    for h in hints:
        txt = font.render(h, True, (140, 150, 170))
        screen.blit(txt, (MAP_WIDTH + 20, y))
        y += 18

def draw_city():
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            tr = traffic_grid[r, c]
            tile_col = COLOR_ROAD
            if tr == 1: tile_col = (58, 55, 42)
            elif tr == 2: tile_col = (72, 38, 38)

            pygame.draw.rect(screen, tile_col, rect)
            pygame.draw.rect(screen, COLOR_GRID_LINE, rect, 1)

            if tr == 1: pygame.draw.circle(screen, COLOR_TRAFFIC_MED, (rect.right - 12, rect.top + 12), 4)
            elif tr == 2: pygame.draw.circle(screen, COLOR_TRAFFIC_JAM, (rect.right - 12, rect.top + 12), 6)

    # Station
    s_rect = pygame.Rect(station_pos[1] * CELL_SIZE + 6, station_pos[0] * CELL_SIZE + 6, CELL_SIZE - 12, CELL_SIZE - 12)
    pygame.draw.rect(screen, (40, 116, 166), s_rect, border_radius=8)
    screen.blit(font.render("GAS", True, COLOR_WHITE), (s_rect.x + 16, s_rect.y + 26))

    # Destinations
    for p in passengers:
        if not p.delivered:
            d_rect = pygame.Rect(p.dest[1] * CELL_SIZE + 8, p.dest[0] * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16)
            pygame.draw.rect(screen, p.color, d_rect, 3, border_radius=8)
            screen.blit(font.render(f"🏁{p.id}", True, p.color), (d_rect.x + 14, d_rect.y + 24))

    # Waiting Passengers
    for p in passengers:
        if not p.picked_up and not p.delivered:
            px = p.pos[1] * CELL_SIZE + CELL_SIZE // 2
            py = p.pos[0] * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, p.color, (px, py), 18)
            screen.blit(font.render(f"P{p.id}", True, (0, 0, 0)), (px - 10, py - 11))

    # Taxis
    for taxi in taxis:
        tx = taxi.pos[1] * CELL_SIZE + 14
        ty = taxi.pos[0] * CELL_SIZE + 14
        t_rect = pygame.Rect(tx, ty, CELL_SIZE - 28, CELL_SIZE - 28)
        pygame.draw.rect(screen, taxi.color, t_rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), t_rect, 2, border_radius=10)

        label = f"T{taxi.id}" if not taxi.passenger else f"{taxi.passenger.id}🚗"
        screen.blit(font.render(label, True, (0, 0, 0)), (tx + 12, ty + 18))

# 3. INTERACTIVE LOOP
running = True

while running:
    curr_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                taxis, passengers = init_world()
            elif event.key == pygame.K_p:
                paused = not paused
            elif event.key == pygame.K_UP:
                speed_delay = max(0.05, speed_delay - 0.05)
            elif event.key == pygame.K_DOWN:
                speed_delay = min(0.8, speed_delay + 0.05)

    if not paused and (curr_time - last_move_time >= speed_delay):
        last_move_time = curr_time
        step_counter += 1

        if step_counter % 8 == 0:
            traffic_grid = np.random.choice([0, 1, 2], size=(GRID_SIZE, GRID_SIZE), p=[0.75, 0.20, 0.05])

        for i, taxi in enumerate(taxis):
            other_taxi = taxis[1 - i]

            # 1. PIT STOP CHECK & RELEASE RESERVATIONS
            if taxi.fuel <= 12 and not taxi.passenger:
                if not taxi.going_to_refuel:
                    taxi.going_to_refuel = True
                    # Release any claimed passenger so the other cab can take them!
                    for p in passengers:
                        if p.assigned_taxi == taxi:
                            p.assigned_taxi = None
                taxi.status = "Low Fuel -> Pit Stop ⛽"
            
            # If at gas station, refuel immediately!
            if taxi.pos == station_pos and taxi.fuel < taxi.max_fuel:
                taxi.fuel = taxi.max_fuel
                taxi.going_to_refuel = False
                taxi.status = "Tank Full! ⛽"

            # 2. TARGET SELECTION
            if taxi.going_to_refuel:
                target_pos = station_pos
                dest_pos = station_pos
            elif taxi.passenger:
                target_pos = taxi.passenger.pos
                dest_pos = taxi.passenger.dest
            else:
                available = [p for p in passengers if not p.picked_up and not p.delivered and (p.assigned_taxi is None or p.assigned_taxi == taxi)]
                if available:
                    available.sort(key=lambda p: abs(taxi.pos[0] - p.pos[0]) + abs(taxi.pos[1] - p.pos[1]))
                    target = available[0]
                    target.assigned_taxi = taxi
                    taxi.status = f"En route to P{target.id}"
                    target_pos = target.pos
                    dest_pos = target.dest
                else:
                    taxi.status = "Idle (Patrolling)"
                    # Wander towards center if idle
                    target_pos = [2, 2]
                    dest_pos = [2, 2]

            # 3. PICKUP
            if not taxi.going_to_refuel and not taxi.passenger and taxi.pos == target_pos:
                taxi.passenger = target
                target.picked_up = True
                taxi.status = f"Delivering P{target.id} 🏁"
                taxi.total_score += 60
                continue

            # 4. DELIVERY & NEW PASSENGER SPAWN
            if taxi.passenger and taxi.pos == dest_pos:
                taxi.completed_rides += 1
                taxi.total_score += 80
                taxi.status = f"P{taxi.passenger.id} Delivered! 🎉"
                
                p_index = 0 if taxi.passenger.id == "A" else 1
                p_color = (241, 196, 15) if taxi.passenger.id == "A" else (230, 126, 34)
                passengers[p_index] = spawn_single_passenger(taxi.passenger.id, p_color)
                taxi.passenger = None
                continue

            # 5. ASK BRAIN FOR ACTION
            obs = get_obs(taxi, target_pos, dest_pos)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                q_vals = brain(state_tensor).squeeze(0)
                action = q_vals[:4].argmax().item()

            # 6. MOVE
            move_taxi(taxi, action, other_taxi)

    screen.fill(COLOR_BG)
    draw_city()
    draw_hud()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()