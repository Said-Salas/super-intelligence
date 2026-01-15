import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
WORLD_SIZE = 50
INITIAL_ENERGY = 150  # More energy to start
FOOD_ENERGY = 40
METABOLISM_COST = 0.5 # Lower cost for smooth movement
SENSOR_ANGLE = np.pi / 4  # Sensors are 45 degrees apart
SENSOR_DIST = 4.0     # How far ahead sensors are
SPEED = 2.0           # Base speed

class Organism:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.energy = INITIAL_ENERGY
        self.alive = True
        self.age = 0
        
        # The Brain: Weights connecting Sensors to Motors
        # [Left_Sensor, Right_Sensor] -> [Left_Motor, Right_Motor]
        # Crossed wiring (Excitory): Strong signal on Left -> Right motor speeds up -> Turns Left
        self.brain_weights = np.array([
            [0.1, 1.0],  # Left Sensor connects to: Left Motor (weak), Right Motor (strong)
            [1.0, 0.1]   # Right Sensor connects to: Left Motor (strong), Right Motor (weak)
        ])

    def get_sensor_positions(self):
        # Calculate where the "antennae" are based on current angle
        left_angle = self.angle + SENSOR_ANGLE
        right_angle = self.angle - SENSOR_ANGLE
        
        left_pos = self.pos + SENSOR_DIST * np.array([np.cos(left_angle), np.sin(left_angle)])
        right_pos = self.pos + SENSOR_DIST * np.array([np.cos(right_angle), np.sin(right_angle)])
        
        return left_pos, right_pos

    def sense(self, food_items):
        if len(food_items) == 0:
            return np.array([0.0, 0.0])

        left_pos, right_pos = self.get_sensor_positions()
        
        # Smell intensity = 1 / distance (Inverse square law simplified)
        # We sum the smell from ALL food, like real smell
        dists_l = np.linalg.norm(food_items - left_pos, axis=1)
        dists_r = np.linalg.norm(food_items - right_pos, axis=1)
        
        # Avoid division by zero with small epsilon
        signal_l = np.sum(1.0 / (dists_l + 0.1))
        signal_r = np.sum(1.0 / (dists_r + 0.1))
        
        # Clip signals to reasonable range (simulating neuron saturation)
        return np.array([min(signal_l, 5.0), min(signal_r, 5.0)])

    def act(self, sensor_input):
        if not self.alive:
            return

        # THE BRAIN: Inputs * Weights = Motor Output
        # This matrix multiplication replaces the "if/else" logic!
        motor_output = np.dot(sensor_input, self.brain_weights)
        
        # Base speed (wandering) + Motor speed (reaction to food)
        left_motor = 0.5 + motor_output[0]
        right_motor = 0.5 + motor_output[1]
        
        # Differential Drive Physics
        # If Right Motor > Left Motor, we turn Left.
        velocity = (left_motor + right_motor) / 2 * SPEED
        rotation = (right_motor - left_motor) * 0.2
        
        # Update State
        self.angle += rotation
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.pos += direction * velocity
        
        # Wrap around world (Torus topology) - Helps survival!
        self.pos = self.pos % WORLD_SIZE
        
        self.energy -= METABOLISM_COST
        self.age += 1
        if self.energy <= 0:
            self.alive = False

class World:
    def __init__(self):
        self.organism = Organism(WORLD_SIZE/2, WORLD_SIZE/2)
        self.food = np.zeros((0, 2))
        self.spawn_food(30) # More food for the new physics

    def spawn_food(self, count):
        new_food = np.random.uniform(0, WORLD_SIZE, (count, 2))
        if len(self.food) > 0:
            self.food = np.vstack((self.food, new_food))
        else:
            self.food = new_food

    def update(self):
        if not self.organism.alive:
            return

        # 1. Sense (Get inputs)
        inputs = self.organism.sense(self.food)
        
        # 2. Act (Process -> Motors)
        self.organism.act(inputs)

        # 3. Eat
        if len(self.food) > 0:
            dists = np.linalg.norm(self.food - self.organism.pos, axis=1)
            # Eating radius
            eaten_indices = np.where(dists < 2.0)[0]
            
            if len(eaten_indices) > 0:
                self.organism.energy += FOOD_ENERGY * len(eaten_indices)
                self.food = np.delete(self.food, eaten_indices, axis=0)
                self.spawn_food(len(eaten_indices))

# --- Visualization ---
world = World()

fig, ax = plt.subplots()
ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_title("Level 2: Embodiment & Wiring")

food_scatter = ax.scatter([], [], c='green', label='Food', s=40, alpha=0.6)
# Organism is now an arrow to show direction
org_plot, = ax.plot([], [], 'b-o', markersize=10, linewidth=2, label='Organism') 
left_sensor_plot, = ax.plot([], [], 'r.', markersize=5)
right_sensor_plot, = ax.plot([], [], 'r.', markersize=5)
energy_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def animate(frame):
    world.update()
    
    # Food
    if len(world.food) > 0:
        food_scatter.set_offsets(world.food)
    else:
        food_scatter.set_offsets(np.empty((0, 2)))

    # Organism Body & Sensors
    l_pos, r_pos = world.organism.get_sensor_positions()
    
    # Draw body line from tail (pos) to head (between sensors)
    head_x = (l_pos[0] + r_pos[0]) / 2
    head_y = (l_pos[1] + r_pos[1]) / 2
    
    org_plot.set_data([world.organism.pos[0], head_x], [world.organism.pos[1], head_y])
    left_sensor_plot.set_data([l_pos[0]], [l_pos[1]])
    right_sensor_plot.set_data([r_pos[0]], [r_pos[1]])
    
    status = "ALIVE" if world.organism.alive else "DEAD"
    color = "blue" if world.organism.alive else "red"
    org_plot.set_color(color)
    
    energy_text.set_text(f"Status: {status}\nEnergy: {world.organism.energy:.1f}\nAge: {world.organism.age}")
    
    return food_scatter, org_plot, left_sensor_plot, right_sensor_plot, energy_text

ani = animation.FuncAnimation(fig, animate, interval=30, blit=False)
plt.legend(loc='lower right')
plt.show()