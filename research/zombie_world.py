import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
WORLD_SIZE = 100
SENSOR_RANGE = 40
NUM_RAYS = 8
FOV = 360 # degrees

# Physics
SPEED = 3.0
TURN_SPEED = 0.3 # Radians

# Learning
LEARNING_RATE = 0.01
GAMMA = 0.9 # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

class DeepBrain:
    def __init__(self, input_size, hidden_size, output_size):
        # Simple 2-Layer Neural Network
        # Weights initialized with He initialization for ReLU
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Memory for backprop
        self.last_input = None
        self.last_hidden = None

    def predict(self, inputs):
        # Forward pass
        # 1. Hidden Layer (ReLU)
        self.last_input = inputs
        self.z1 = np.dot(inputs, self.w1) + self.b1
        self.last_hidden = np.maximum(0, self.z1) # ReLU
        
        # 2. Output Layer (Linear for Q-values)
        self.z2 = np.dot(self.last_hidden, self.w2) + self.b2
        return self.z2

    def train(self, state, action_idx, target_q):
        # We only train on the specific action taken
        # Target Q is the "ground truth" Q-value we wanted for that action
        
        # 1. Forward pass (to get current predictions)
        current_q_values = self.predict(state)
        
        # 2. Calculate Error for the specific action
        # We want predicted[action] to be closer to target_q
        current_val = current_q_values[0, action_idx]
        error = current_val - target_q
        
        # 3. Backpropagation
        
        # Gradient of Output Layer
        # dError/dOutput is 0 for non-chosen actions, and 1*error for chosen
        d_output = np.zeros_like(current_q_values)
        d_output[0, action_idx] = error 
        
        # Gradient of W2
        d_w2 = np.dot(self.last_hidden.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient of Hidden Layer
        d_hidden = np.dot(d_output, self.w2.T)
        d_hidden[self.z1 <= 0] = 0 # ReLU derivative
        
        # Gradient of W1
        d_w1 = np.dot(self.last_input.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update Weights (Gradient Descent)
        self.w1 -= LEARNING_RATE * d_w1
        self.b1 -= LEARNING_RATE * d_b1
        self.w2 -= LEARNING_RATE * d_w2
        self.b2 -= LEARNING_RATE * d_b2
        
        return abs(error)

class Agent:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.angle = np.random.uniform(0, 2 * np.pi)
        
        # Brain: 16 Inputs (8 rays * 2 types), 16 Hidden, 3 Actions
        self.brain = DeepBrain(16, 24, 3)
        
        self.epsilon = EPSILON_START
        self.alive = True
        self.age = 0
        self.total_reward = 0
        
        # Memory for Experience Replay (Simplified: just last step)
        self.last_state = None
        self.last_action = None

    def get_sensors(self, food_list, zombie_list):
        # 8 Rays distributed around 360 degrees
        # Returns: [Food_Dist_0, Zombie_Dist_0, Food_Dist_1, ...]
        
        readings = []
        
        # Calculate relative vectors to all objects once
        # This is a vectorized optimization instead of ray-casting physics
        
        # Pre-process objects into relative polar coordinates (dist, angle)
        def get_relative(objects):
            if len(objects) == 0: return np.array([]), np.array([])
            vecs = objects - self.pos
            dists = np.linalg.norm(vecs, axis=1)
            angles = np.arctan2(vecs[:, 1], vecs[:, 0]) - self.angle
            # Normalize angles to -pi to pi
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            return dists, angles

        f_dists, f_angles = get_relative(food_list)
        z_dists, z_angles = get_relative(zombie_list)
        
        # For each ray sector
        for i in range(NUM_RAYS):
            ray_angle = (i / NUM_RAYS) * 2 * np.pi - np.pi # -pi to pi distribution
            
            # Simple sector logic: Object is "seen" by ray if it's within angular slice
            sector_width = (2 * np.pi) / NUM_RAYS
            
            # Distance defaults to 1.0 (far away) if nothing seen
            # We invert distance: 0 = far/none, 1 = close
            f_val = 0
            z_val = 0
            
            # Check Food
            if len(f_dists) > 0:
                # Find food in this sector
                angle_diffs = np.abs(f_angles - ray_angle)
                angle_diffs[angle_diffs > np.pi] = 2*np.pi - angle_diffs[angle_diffs > np.pi] # Wrap
                
                matches = (angle_diffs < sector_width/2) & (f_dists < SENSOR_RANGE)
                if np.any(matches):
                    # Closest match
                    closest = np.min(f_dists[matches])
                    f_val = 1.0 - (closest / SENSOR_RANGE)
            
            # Check Zombies
            if len(z_dists) > 0:
                angle_diffs = np.abs(z_angles - ray_angle)
                angle_diffs[angle_diffs > np.pi] = 2*np.pi - angle_diffs[angle_diffs > np.pi]
                
                matches = (angle_diffs < sector_width/2) & (z_dists < SENSOR_RANGE)
                if np.any(matches):
                    closest = np.min(z_dists[matches])
                    z_val = 1.0 - (closest / SENSOR_RANGE)
            
            readings.extend([f_val, z_val])
            
        return np.array(readings).reshape(1, -1)

    def act(self, sensors):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 3) # Explore
        else:
            q_values = self.brain.predict(sensors)
            action = np.argmax(q_values) # Exploit
        
        self.last_state = sensors
        self.last_action = action
        return action

    def update(self, action, world_size):
        # Physics Update
        # Actions: 0=Left, 1=Forward, 2=Right
        if action == 0:
            self.angle += TURN_SPEED
        elif action == 2:
            self.angle -= TURN_SPEED
            
        # Move forward (always move a bit, turning just steers)
        # Or maybe only move on Forward? Let's say moves on all, but slower on turns?
        # Plan says: Turn Left, Move Forward, Turn Right. Implies distinct.
        speed = SPEED if action == 1 else SPEED * 0.5
        
        self.pos[0] += np.cos(self.angle) * speed
        self.pos[1] += np.sin(self.angle) * speed
        
        # Walls (Bounce/Clip)
        self.pos = np.clip(self.pos, 0, world_size)
        self.age += 1
        
        # Epsilon Decay
        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

    def learn(self, reward, next_sensors, done):
        if self.last_state is None: return 0
        
        # Q-Learning Target
        # Target = Reward + Gamma * Max(Q(next_state))
        if done:
            target = reward
        else:
            next_q = self.brain.predict(next_sensors)
            target = reward + GAMMA * np.max(next_q)
            
        # Train brain
        loss = self.brain.train(self.last_state, self.last_action, target)
        return loss

# --- WORLD & ANIMATION ---

# Globals
agent = Agent(WORLD_SIZE/2, WORLD_SIZE/2)
food_items = np.random.uniform(5, WORLD_SIZE-5, (10, 2))
zombies = np.random.uniform(5, WORLD_SIZE-5, (5, 2))
# Give zombies random directions
zombie_angles = np.random.uniform(0, 2*np.pi, 5)

history_score = []
history_loss = []

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 2)

# Map Plot
ax_map = fig.add_subplot(gs[:, 0])
ax_map.set_xlim(0, WORLD_SIZE)
ax_map.set_ylim(0, WORLD_SIZE)
ax_map.set_title("Deep Zombie World")

agent_dot, = ax_map.plot([], [], 'bo', ms=10, label='Agent')
agent_dir, = ax_map.plot([], [], 'b-', lw=2)
food_dots, = ax_map.plot([], [], 'go', ms=8, label='Food')
zombie_dots, = ax_map.plot([], [], 'ro', ms=8, label='Zombie')

# Sensors (Visualization)
sensor_lines = []
for _ in range(NUM_RAYS):
    line, = ax_map.plot([], [], 'k-', alpha=0.2)
    sensor_lines.append(line)

# Score Plot
ax_score = fig.add_subplot(gs[0, 1])
ax_score.set_title("Total Reward")
line_score, = ax_score.plot([], [], 'g-')

# Loss Plot
ax_loss = fig.add_subplot(gs[1, 1])
ax_loss.set_title("Brain Confusion (Loss)")
line_loss, = ax_loss.plot([], [], 'r-')

def animate(frame):
    global food_items, zombies, zombie_angles
    
    # 1. Get Current State
    sensors = agent.get_sensors(food_items, zombies)
    
    # 2. Decide Action
    action = agent.act(sensors)
    
    # 3. Execute Action
    agent.update(action, WORLD_SIZE)
    
    # 4. Check Collisions & Rewards
    reward = -1 # Living cost
    done = False
    
    # Food Collision
    # Distances to all food
    f_dists = np.linalg.norm(food_items - agent.pos, axis=1)
    eaten = f_dists < 4.0
    if np.any(eaten):
        n_eaten = np.sum(eaten)
        reward += 15 * n_eaten
        # Respawn eaten food
        food_items[eaten] = np.random.uniform(5, WORLD_SIZE-5, (n_eaten, 2))
    
    # Zombie Collision
    z_dists = np.linalg.norm(zombies - agent.pos, axis=1)
    if np.any(z_dists < 4.0):
        reward -= 50
        # Teleport away from zombie slightly to prevent instant death loop
        # Or simple "game over" reset? Let's just punish and bump.
        agent.pos = np.array([WORLD_SIZE/2, WORLD_SIZE/2])
        done = True # "Episode" ended effectively
        
    agent.total_reward += reward
    
    # 5. Learn
    # We need the state AFTER the move to calculate target
    next_sensors = agent.get_sensors(food_items, zombies)
    loss = agent.learn(reward, next_sensors, done)
    
    # 6. Zombie Logic (Random Walk)
    zombie_angles += np.random.uniform(-0.2, 0.2, len(zombies))
    zombie_vecs = np.column_stack((np.cos(zombie_angles), np.sin(zombie_angles))) * 1.0 # Slow zombies
    zombies += zombie_vecs
    zombies = np.clip(zombies, 0, WORLD_SIZE)
    # Bounce angles
    for i in range(len(zombies)):
        if zombies[i, 0] <= 0 or zombies[i, 0] >= WORLD_SIZE: zombie_angles[i] = np.pi - zombie_angles[i]
        if zombies[i, 1] <= 0 or zombies[i, 1] >= WORLD_SIZE: zombie_angles[i] = -zombie_angles[i]

    # --- VISUALIZATION UPDATES ---
    
    # Agent Body
    agent_dot.set_data([agent.pos[0]], [agent.pos[1]])
    # Agent Head Direction
    head_x = agent.pos[0] + np.cos(agent.angle) * 5
    head_y = agent.pos[1] + np.sin(agent.angle) * 5
    agent_dir.set_data([agent.pos[0], head_x], [agent.pos[1], head_y])
    
    # Objects
    food_dots.set_data(food_items[:, 0], food_items[:, 1])
    zombie_dots.set_data(zombies[:, 0], zombies[:, 1])
    
    # Sensor Rays (Visual)
    # Only draw if we want (it's expensive). Let's draw loosely.
    for i, line in enumerate(sensor_lines):
        ray_angle = agent.angle + (i / NUM_RAYS) * 2 * np.pi - np.pi
        rx = agent.pos[0] + np.cos(ray_angle) * SENSOR_RANGE
        ry = agent.pos[1] + np.sin(ray_angle) * SENSOR_RANGE
        line.set_data([agent.pos[0], rx], [agent.pos[1], ry])
        
        # Color line if it sees something (using the sensor reading)
        # sensors is shape (1, 16) -> [f0, z0, f1, z1...]
        f_val = sensors[0, i*2]
        z_val = sensors[0, i*2+1]
        if z_val > 0: line.set_color('red'); line.set_alpha(0.5)
        elif f_val > 0: line.set_color('green'); line.set_alpha(0.5)
        else: line.set_color('black'); line.set_alpha(0.1)

    # Graphs
    history_score.append(agent.total_reward)
    history_loss.append(loss if loss is not None else 0)
    
    # Keep history manageable
    if len(history_score) > 200:
        history_score.pop(0)
        history_loss.pop(0)
        
    line_score.set_data(range(len(history_score)), history_score)
    ax_score.set_xlim(0, len(history_score))
    ax_score.set_ylim(min(history_score)-10, max(history_score)+10)
    
    line_loss.set_data(range(len(history_loss)), history_loss)
    ax_loss.set_xlim(0, len(history_loss))
    ax_loss.set_ylim(0, max(history_loss) + 0.1)

    return agent_dot, agent_dir, food_dots, zombie_dots, *sensor_lines, line_score, line_loss

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
plt.tight_layout()
plt.show()

