import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
WIDTH, HEIGHT = 100, 100
INITIAL_BACTERIA = 10
FOOD_SPAWN_RATE = 2  # New food per frame
BACTERIA_SPEED = 1.0
ENERGY_TO_REPRODUCE = 50
STARTING_ENERGY = 20
ENERGY_LOSS_PER_TICK = 0.2
FOOD_ENERGY = 15

class Bacterium:
    def __init__(self, x, y, energy):
        self.x = x
        self.y = y
        self.energy = energy
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.age = 0
        
    def move(self, food_list):
        # 1. Simple sensing: Find nearest food
        nearest_food = None
        min_dist = np.inf
        
        # Look for food within a small radius
        sensor_radius = 15
        for f in food_list:
            dist = np.hypot(f[0] - self.x, f[1] - self.y)
            if dist < min_dist and dist < sensor_radius:
                min_dist = dist
                nearest_food = f

        # 2. Behavior: Steer towards food or wander
        if nearest_food:
            target_angle = np.arctan2(nearest_food[1] - self.y, nearest_food[0] - self.x)
            # Smoothly turn towards target
            self.angle = 0.8 * self.angle + 0.2 * target_angle
        else:
            # Random "brownian" motion / wandering
            self.angle += np.random.uniform(-0.5, 0.5)

        # 3. Update Position
        self.x += np.cos(self.angle) * BACTERIA_SPEED
        self.y += np.sin(self.angle) * BACTERIA_SPEED
        
        # 4. Boundary wrapping (toroidal world)
        self.x %= WIDTH
        self.y %= HEIGHT
        
        # 5. Metabolism
        self.energy -= ENERGY_LOSS_PER_TICK
        self.age += 1

class World:
    def __init__(self):
        self.bacteria = [Bacterium(np.random.rand()*WIDTH, np.random.rand()*HEIGHT, STARTING_ENERGY) 
                         for _ in range(INITIAL_BACTERIA)]
        self.food = [] # List of [x, y] coordinates
        
    def update(self):
        # Spawn food randomly
        for _ in range(FOOD_SPAWN_RATE):
            self.food.append([np.random.rand()*WIDTH, np.random.rand()*HEIGHT])
            
        # Update each bacterium
        new_bacteria = []
        for b in self.bacteria:
            b.move(self.food)
            
            # Eat food
            # Filter out eaten food (simple collision detection)
            # Creating a new list is inefficient for large scale, but simple for this demo
            surviving_food = []
            eaten = False
            for f in self.food:
                dist = np.hypot(f[0] - b.x, f[1] - b.y)
                if dist < 2.0 and not eaten: # Eat if close enough
                    b.energy += FOOD_ENERGY
                    eaten = True # Can only eat one piece per tick
                else:
                    surviving_food.append(f)
            self.food = surviving_food

            # Reproduction
            if b.energy > ENERGY_TO_REPRODUCE:
                b.energy /= 2
                # Create offspring at same location
                offspring = Bacterium(b.x, b.y, b.energy)
                offspring.angle = b.angle + np.pi # Swim away from parent
                new_bacteria.append(offspring)
            
            # Death
            if b.energy > 0:
                new_bacteria.append(b)
        
        self.bacteria = new_bacteria

# --- VISUALIZATION ---
world = World()

fig, ax = plt.subplots()
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title("Digital Bacteria Simulation")

# Graphics objects
bacteria_scatter = ax.scatter([], [], c='green', s=30, label='Bacteria')
food_scatter = ax.scatter([], [], c='red', s=10, marker='.', label='Food')
text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def animate(frame):
    world.update()
    
    # Update visuals
    if world.bacteria:
        bx = [b.x for b in world.bacteria]
        by = [b.y for b in world.bacteria]
        # Color based on energy (darker = more energy)
        colors = ['#00%02x00' % int(min(255, 100 + b.energy*2)) for b in world.bacteria]
        bacteria_scatter.set_offsets(np.c_[bx, by])
        bacteria_scatter.set_color(colors)
    else:
        bacteria_scatter.set_offsets(np.zeros((0, 2)))

    if world.food:
        fx = [f[0] for f in world.food]
        fy = [f[1] for f in world.food]
        food_scatter.set_offsets(np.c_[fx, fy])
    else:
        food_scatter.set_offsets(np.zeros((0, 2)))

    text_info.set_text(f'Population: {len(world.bacteria)}')
    return bacteria_scatter, food_scatter, text_info

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.legend(loc='lower right')
plt.show()