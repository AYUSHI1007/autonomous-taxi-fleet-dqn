import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MAP_CACHE_FILE = "manhattan_map.graphml"

# 1. Load Manhattan street network
print("📁 Loading real Manhattan street network...")
if os.path.exists(MAP_CACHE_FILE):
    graph = ox.load_graphml(MAP_CACHE_FILE)
else:
    times_square_gps = (40.7580, -73.9855)
    graph = ox.graph_from_point(times_square_gps, dist=450, network_type="drive")
    graph = ox.project_graph(graph)
    ox.save_graphml(graph, MAP_CACHE_FILE)

nodes = list(graph.nodes)
print(f"✅ Manhattan Map Ready: {len(nodes)} Intersections | {len(graph.edges)} Streets")

# 2. Assign Traffic Conditions
TRAFFIC_COLORS = {
    "jam": "#e74c3c",     # Glowing Red (5 km/h)
    "medium": "#e67e22",  # Orange (18 km/h)
    "clear": "#27ae60"    # Green (45 km/h)
}

for u, v, data in graph.edges(data=True):
    length = float(data.get("length", 80.0))
    rand_val = random.random()
    
    if rand_val < 0.22:
        traffic_type = "jam"
        speed = 8.0
    elif rand_val < 0.48:
        traffic_type = "medium"
        speed = 20.0
    else:
        traffic_type = "clear"
        speed = 40.0
        
    data["traffic_type"] = traffic_type
    data["speed_kph"] = speed
    data["traffic_color"] = TRAFFIC_COLORS[traffic_type]
    data["travel_time"] = length / (speed * 1000.0 / 3600.0)

# 3. Setup Matplotlib Window
plt.ion()
fig, ax = plt.subplots(figsize=(10, 9), facecolor="#181a20")

# Civilian traffic cars
civilian_cars = []
for _ in range(12):
    n1 = random.choice(nodes)
    neighbors = list(graph.neighbors(n1))
    if neighbors:
        civilian_cars.append({"u": n1, "v": random.choice(neighbors), "progress": random.random()})

def draw_traffic_streets():
    ax.clear()
    ax.set_facecolor("#181a20")
    ax.axis("off")
    for u, v, data in graph.edges(data=True):
        ux, uy = float(graph.nodes[u]['x']), float(graph.nodes[u]['y'])
        vx, vy = float(graph.nodes[v]['x']), float(graph.nodes[v]['y'])
        col = data.get("traffic_color", "#27ae60")
        linewidth = 4.5 if data.get("traffic_type") == "jam" else 3.5
        ax.plot([ux, vx], [uy, vy], color=col, linewidth=linewidth, alpha=0.85, zorder=1)

def get_waypoint_speed(n1, n2):
    if graph.has_edge(n1, n2):
        edge_data = graph.get_edge_data(n1, n2)[0]
        return edge_data.get("speed_kph", 30.0), edge_data.get("traffic_type", "clear")
    return 30.0, "clear"

def create_smooth_path(node_list, steps_per_segment=10):
    points = []
    speeds = []
    types = []
    is_intersection = []
    
    for i in range(len(node_list) - 1):
        n1 = node_list[i]
        n2 = node_list[i + 1]
        x1, y1 = float(graph.nodes[n1]['x']), float(graph.nodes[n1]['y'])
        x2, y2 = float(graph.nodes[n2]['x']), float(graph.nodes[n2]['y'])
        spd, t_type = get_waypoint_speed(n1, n2)
        
        for step_idx, t in enumerate(np.linspace(0, 1, steps_per_segment, endpoint=False)):
            points.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
            speeds.append(spd)
            types.append(t_type)
            is_intersection.append(step_idx == 0) # True right when hitting an intersection
            
    last_x, last_y = float(graph.nodes[node_list[-1]]['x']), float(graph.nodes[node_list[-1]]['y'])
    points.append((last_x, last_y))
    speeds.append(30.0)
    types.append("clear")
    is_intersection.append(True)
    return points, speeds, types, is_intersection

current_taxi_node = random.choice(nodes)
print("🐢 Slow-Motion Inspection Mode Active! Watch every corner and turn.\n")

CAMERA_RADIUS = 150
trip_num = 1

while plt.fignum_exists(fig.number):
    passenger_node = random.choice(nodes)
    while passenger_node == current_taxi_node:
        passenger_node = random.choice(nodes)

    destination_node = random.choice(nodes)
    while destination_node in [current_taxi_node, passenger_node]:
        destination_node = random.choice(nodes)

    try:
        pickup_nodes = nx.shortest_path(graph, source=current_taxi_node, target=passenger_node, weight="travel_time")
        delivery_nodes = nx.shortest_path(graph, source=passenger_node, target=destination_node, weight="travel_time")
    except nx.NetworkXNoPath:
        current_taxi_node = random.choice(nodes)
        continue

    draw_traffic_streets()

    # Draw Planned Gold Route Line
    full_nodes = pickup_nodes + delivery_nodes[1:]
    rx = [float(graph.nodes[n]['x']) for n in full_nodes]
    ry = [float(graph.nodes[n]['y']) for n in full_nodes]
    ax.plot(rx, ry, color="#f1c40f", linewidth=5.5, alpha=0.9, zorder=2, label="Calculated Route")

    # Civilian Traffic Cars
    civ_x, civ_y = [], []
    for c in civilian_cars:
        ux, uy = float(graph.nodes[c['u']]['x']), float(graph.nodes[c['u']]['y'])
        vx, vy = float(graph.nodes[c['v']]['x']), float(graph.nodes[c['v']]['y'])
        civ_x.append(ux + c['progress'] * (vx - ux))
        civ_y.append(uy + c['progress'] * (vy - uy))
    ax.scatter(civ_x, civ_y, c="#95a5a6", s=60, edgecolors="white", linewidths=1.0, zorder=3, label="Civilian Traffic")

    # Passenger and Destination Markers
    px, py = float(graph.nodes[passenger_node]['x']), float(graph.nodes[passenger_node]['y'])
    dx, dy = float(graph.nodes[destination_node]['x']), float(graph.nodes[destination_node]['y'])
    pass_scatter = ax.scatter([px], [py], c="#2ecc71", s=280, edgecolors="white", linewidths=2.0, zorder=5, label="Passenger")
    dest_scatter = ax.scatter([dx], [dy], c="#e74c3c", s=280, edgecolors="white", linewidths=2.0, zorder=5, label="Destination")

    # Moving Taxi
    start_x, start_y = float(graph.nodes[current_taxi_node]['x']), float(graph.nodes[current_taxi_node]['y'])
    taxi_scatter = ax.scatter([start_x], [start_y], c="#f1c40f", s=450, edgecolors="black", linewidths=3.0, zorder=7, label="AI Taxi")

    ax.legend(facecolor="#21242d", edgecolor="white", labelcolor="white", loc="upper right")

    # --- LEG 1: DRIVE TO PASSENGER IN SLOW MOTION ---
    pickup_waypoints, p_speeds, p_types, p_turns = create_smooth_path(pickup_nodes, steps_per_segment=10)
    for i, (x, y) in enumerate(pickup_waypoints):
        if not plt.fignum_exists(fig.number): break
        taxi_scatter.set_offsets([[x, y]])

        # Camera follow
        ax.set_xlim(x - CAMERA_RADIUS, x + CAMERA_RADIUS)
        ax.set_ylim(y - CAMERA_RADIUS, y + CAMERA_RADIUS)

        cur_speed = p_speeds[i]
        cur_type = p_types[i]
        
        status_label = "CRUISING 🟢" if cur_type == "clear" else ("MODERATE 🟠" if cur_type == "medium" else "RED JAM 🔴")
        dist_rem = int(np.hypot(px - x, py - y))
        
        # Turn notification
        turn_notice = " [📍 TURNING CORNER]" if p_turns[i] else ""
        ax.set_title(f"Trip #{trip_num} | {status_label}{turn_notice} | SPEED: {int(cur_speed)} KM/H | TO PICKUP: {dist_rem}m", 
                     color="white", fontsize=11, weight="bold", pad=12)
        fig.canvas.flush_events()
        
        # SLOW-MOTION PACE: Comfortable delay to inspect each move
        delay = 0.18 if cur_type == "jam" else (0.12 if cur_type == "medium" else 0.08)
        if p_turns[i]:
            delay += 0.25 # Extra pause at intersections so you see the turn!
        time.sleep(delay)

    # --- PICKUP MOMENT ---
    ax.set_title(f"Trip #{trip_num} | 🧍 PASSENGER BOARDED! HEADING TO DESTINATION...", color="#2ecc71", fontsize=12, weight="bold", pad=12)
    pass_scatter.set_visible(False)
    fig.canvas.flush_events()
    time.sleep(1.2)

    # --- LEG 2: DRIVE TO DESTINATION IN SLOW MOTION ---
    delivery_waypoints, d_speeds, d_types, d_turns = create_smooth_path(delivery_nodes, steps_per_segment=10)
    for i, (x, y) in enumerate(delivery_waypoints):
        if not plt.fignum_exists(fig.number): break
        taxi_scatter.set_offsets([[x, y]])

        ax.set_xlim(x - CAMERA_RADIUS, x + CAMERA_RADIUS)
        ax.set_ylim(y - CAMERA_RADIUS, y + CAMERA_RADIUS)

        cur_speed = d_speeds[i]
        cur_type = d_types[i]
        
        status_label = "CRUISING 🟢" if cur_type == "clear" else ("MODERATE 🟠" if cur_type == "medium" else "RED JAM 🔴")
        dist_rem = int(np.hypot(dx - x, dy - y))
        turn_notice = " [📍 TURNING CORNER]" if d_turns[i] else ""
        
        ax.set_title(f"Trip #{trip_num} | {status_label}{turn_notice} | SPEED: {int(cur_speed)} KM/H | TO GOAL: {dist_rem}m", 
                     color="white", fontsize=11, weight="bold", pad=12)
        fig.canvas.flush_events()
        
        delay = 0.18 if cur_type == "jam" else (0.12 if cur_type == "medium" else 0.08)
        if d_turns[i]:
            delay += 0.25
        time.sleep(delay)

    # --- ARRIVAL CELEBRATION ---
    ax.set_title(f"Trip #{trip_num} | 🎉 RIDE COMPLETED! DELIVERED IN MANHATTAN!", color="#f1c40f", fontsize=12, weight="bold", pad=12)
    fig.canvas.flush_events()
    time.sleep(2.0)

    current_taxi_node = destination_node
    trip_num += 1

plt.ioff()