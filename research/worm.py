import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
WORLD_SIZE = 50
INITIAL_ENERGY = 100
FOOD_ENERGY = 30
METABOLISM_COST = 1.0
SENSE_RADIUS = 5

class Organism:
    def __init__(self, x, y):
        self.pos = np.array([float(x), float(y)])
        self.energy = INITIAL_ENERGY
        self.alive = True
        self.age = 0

    def sense(self, food_items):
        # FIX: Check length instead of truthiness for numpy arrays
        if len(food_items) == 0:
            return None
        
        distances = np.linalg.norm(food_items - self.pos, axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist <= SENSE_RADIUS:
            return food_items[min_dist_idx]
        return None

    def act(self, target):
        if not self.alive:
            return

        move_vector = np.array([0.0, 0.0])

        if target is not None:
            direction = target - self.pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                move_vector = direction / norm
        else:
            move_vector = np.random.randn(2)
            norm = np.linalg.norm(move_vector)
            if norm > 0:
                move_vector = move_vector / norm

        self.pos += move_vector
        self.pos = np.clip(self.pos, 0, WORLD_SIZE)
        self.energy -= METABOLISM_COST
        self.age += 1

        if self.energy <= 0:
            self.alive = False

class World:
    def __init__(self):
        self.organism = Organism(WORLD_SIZE/2, WORLD_SIZE/2)
        # FIX: Initialize as empty numpy array with correct shape (0, 2)
        self.food = np.zeros((0, 2)) 
        self.spawn_food(20)

    def spawn_food(self, count):
        # FIX: Generate new food and stack it correctly
        new_food = np.random.uniform(0, WORLD_SIZE, (count, 2))
        if len(self.food) > 0:
            self.food = np.vstack((self.food, new_food))
        else:
            self.food = new_food

    def update(self):
        if not self.organism.alive:
            return

        target = self.organism.sense(self.food)
        self.organism.act(target)

        if len(self.food) > 0:
            dists = np.linalg.norm(self.food - self.organism.pos, axis=1)
            eaten_indices = np.where(dists < 1.0)[0]
            
            if len(eaten_indices) > 0:
                self.organism.energy += FOOD_ENERGY * len(eaten_indices)
                self.food = np.delete(self.food, eaten_indices, axis=0)
                self.spawn_food(len(eaten_indices))

# --- Visualization ---
world = World()

fig, ax = plt.subplots()
ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_title("Proto-Intelligence: Homeostasis Loop")

food_scatter = ax.scatter([], [], c='green', label='Food', s=50)
org_scatter = ax.scatter([], [], c='blue', label='Organism', s=100)
energy_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Initialize with starting positions so they appear immediately
if len(world.food) > 0:
    food_scatter.set_offsets(world.food)
org_scatter.set_offsets([world.organism.pos])

def animate(frame):
    world.update()
    
    if len(world.food) > 0:
        food_scatter.set_offsets(world.food)
    else:
        food_scatter.set_offsets(np.empty((0, 2)))

    org_scatter.set_offsets([world.organism.pos])
    
    status = "ALIVE" if world.organism.alive else "DEAD"
    color = "blue" if world.organism.alive else "red"
    org_scatter.set_color(color)
    
    energy_text.set_text(f"Status: {status}\nEnergy: {world.organism.energy:.1f}\nAge: {world.organism.age}")
    
    return food_scatter, org_scatter, energy_text

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.legend()
plt.show()