from ursina import *
import random
import time
import json
import os
import math

# --- THE ASSOCIATIVE MEMORY (The Real Brain) ---
class AssociativeBrain:
    def __init__(self, memory_file="cortex_memory.json"):
        self.memory_file = memory_file
        self.short_term_memory = [] # Buffer for (State, Action) trace
        self.long_term_memory = {}  # The "Knowledge" {State_Hash: [Q_Forward, Q_Turn]}
        self.epsilon = 0.2 # Curiosity (20% chance to try random stuff)
        self.last_action = 0 # 0=Forward, 1=Turn
        self.last_state = None
        
        # Load Memory if exists (Persistence)
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.long_term_memory = json.load(f)
            print(f"--- BRAIN LOADED: {len(self.long_term_memory)} Memories ---")

    def get_state_signature(self, input_vector):
        """
        Compresses reality into a 'Simplicity Hash'.
        Input: [R_signal, G_signal, Obstacle_Dist]
        Output: A string key like "R1_G0_D5"
        """
        # Discretize values to group similar situations
        r = int(input_vector['red_signal'] > 0)  # 0 or 1
        g = int(input_vector['green_signal'] > 0) # 0 or 1
        d = int(input_vector['center_dist'] / 5) # 0, 1, 2, 3 (Distance buckets)
        
        return f"R{r}_G{g}_D{d}"

    def think(self, input_vector):
        state = self.get_state_signature(input_vector)
        self.last_state = state
        
        # Initialize state if new
        if state not in self.long_term_memory:
            self.long_term_memory[state] = [0.0, 0.0] # [Value_Forward, Value_Turn]
        
        # Decide: Explore or Exploit?
        if random.random() < self.epsilon:
            action = random.choice([0, 1])
            thought = "Exploring..."
        else:
            # Pick best known action
            values = self.long_term_memory[state]
            if values[0] > values[1]:
                action = 0 # Forward
                thought = "I know this: Move Forward"
            elif values[1] > values[0]:
                action = 1 # Turn
                thought = "I know this: Turn"
            else:
                action = random.choice([0, 1])
                thought = "Unsure..."

        self.last_action = action
        return action, thought

    def learn(self, reward):
        """
        The Core Loop: Result -> Update Previous State
        """
        if self.last_state:
            # Q-Learning Update Rule (Simplified)
            # Old_Value += Learning_Rate * (Reward - Old_Value)
            current_val = self.long_term_memory[self.last_state][self.last_action]
            new_val = current_val + 0.5 * (reward - current_val)
            self.long_term_memory[self.last_state][self.last_action] = new_val
            
            # Auto-Save (The "Continuous Loop")
            # In production, save less often to save disk IO
            if random.random() < 0.05: 
                self.save()

    def save(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.long_term_memory, f)
        print("--- MEMORY SAVED ---")

# --- THE RETINA (System 1) ---
class Retina:
    def __init__(self, camera_entity):
        self.cam = camera_entity
        
    def process_frame(self):
        # Raycast Scan (Fast Lidar)
        red_mass = 0
        green_mass = 0
        center_obstacle = 20.0
        
        # Wide Scan
        for i in range(-5, 6):
            angle = i * 5
            rad = math.radians(self.cam.world_rotation_y + angle)
            direction = Vec3(math.sin(rad), 0, math.cos(rad))
            hit = raycast(self.cam.world_position, direction, distance=20, ignore=(agent,))
            
            if hit.hit:
                if i == 0: center_obstacle = hit.distance
                if hit.entity.color == color.red: red_mass += 1
                elif hit.entity.color == color.green: green_mass += 1
        
        return {
            "red_signal": red_mass,
            "green_signal": green_mass,
            "center_dist": center_obstacle
        }

# --- THE WORLD ---
app = Ursina()

# Environment
ground = Entity(model='plane', scale=(50, 1, 50), color=color.gray.tint(-.2), texture='white_cube', texture_scale=(50,50), collider='box')
walls = [
    Entity(model='cube', position=(0,2.5,25), scale=(50,5,1), color=color.gray, collider='box'),
    Entity(model='cube', position=(0,2.5,-25), scale=(50,5,1), color=color.gray, collider='box'),
    Entity(model='cube', position=(25,2.5,0), scale=(1,5,50), color=color.gray, collider='box'),
    Entity(model='cube', position=(-25,2.5,0), scale=(1,5,50), color=color.gray, collider='box')
]
ball = Entity(model='sphere', color=color.red, scale=(2,2,2), position=(5, 1, 10), collider='sphere')
cube = Entity(model='cube', color=color.green, scale=(2,2,2), position=(-5, 1, 10), collider='box')

agent = Entity(model='capsule', color=color.white, scale=(1,2,1), position=(0, 1, 0), collider='capsule')
head = Entity(parent=agent, position=(0, 0.8, 0))

# The Components
retina = Retina(agent)
brain = AssociativeBrain()

# GUI
thought_bubble = Text(text="Init...", position=(-0.85, 0.45), scale=1.5, background=True)

def update():
    # 1. SENSE
    sensation = retina.process_frame()
    
    # 2. THINK
    action, thought = brain.think(sensation)
    thought_bubble.text = f"{thought}\nState: {brain.last_state}"
    
    # 3. ACT
    speed = 10 * time.dt
    turn_speed = 150 * time.dt
    
    if action == 0: # Forward
        agent.position += agent.forward * speed
    else: # Turn
        agent.rotation_y += turn_speed
        
    # 4. PHYSICS & REWARD (The Teacher)
    # Wall Punishment
    if sensation['center_dist'] < 2:
        brain.learn(-1.0) # Pain
        agent.position -= agent.forward * speed * 2 # Bounce back
        agent.rotation_y += 180 # Turn around
        
    # Ball Reward
    if distance(agent, ball) < 2:
        brain.learn(5.0) # Pleasure
        print("ATE RED BALL!")
        ball.x = random.uniform(-20, 20) # Respawn
        ball.z = random.uniform(-20, 20)

    # Cube Reward
    if distance(agent, cube) < 2:
        brain.learn(5.0)
        print("ATE GREEN CUBE!")
        cube.x = random.uniform(-20, 20)
        cube.z = random.uniform(-20, 20)

    # Bounds
    agent.x = clamp(agent.x, -24, 24)
    agent.z = clamp(agent.z, -24, 24)

# Camera
EditorCamera()
camera.position = (0, 20, -20)
camera.look_at(agent)

app.run()
