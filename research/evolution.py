import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- GOD MODE CONFIGURATION ---
WIDTH, HEIGHT = 100, 100
INIT_POPULATION = 20
PLANT_GROWTH_RATE = 3
MUTATION_RATE = 0.1

# Physics Constants
BASE_ENERGY_LOSS = 0.1
SPEED_COST = 0.05
SIZE_COST = 0.02

class Organism:
    def __init__(self, x, y, dna=None, energy=50):
        self.x = x
        self.y = y
        self.energy = energy
        self.age = 0
        
        # DNA: [Speed (0-1), Size (0-1), Diet (0-1)]
        # Diet: 0 = Herbivore (Green), 1 = Carnivore (Red)
        if dna is None:
            self.dna = np.random.rand(3)
        else:
            # Mutation: slightly tweak the parent's DNA
            mutation = np.random.normal(0, 0.1, 3)
            self.dna = np.clip(dna + mutation, 0, 1)
            
        # Extract traits for readability
        self.speed_trait = 0.5 + self.dna[0] * 1.5 # Range: 0.5 - 2.0
        self.size_trait = 5 + self.dna[1] * 10     # Range: 5 - 15
        self.is_carnivore = self.dna[2] > 0.6      # Threshold to be a killer
        
        # Color based on Diet
        # Herbivores = Blue/Cyan, Carnivores = Red/Orange
        if self.is_carnivore:
            self.color = (1.0, 1.0 - self.dna[0], 0) # Redder = Faster
        else:
            self.color = (0, self.dna[0], 1.0)       # Bluer = Faster

    def update(self, plants, others):
        # 1. MOVE
        # Wander with some direction changes
        angle = np.random.uniform(0, 2*np.pi)
        self.x += np.cos(angle) * self.speed_trait
        self.y += np.sin(angle) * self.speed_trait
        
        # Wrap world
        self.x %= WIDTH
        self.y %= HEIGHT
        
        # 2. METABOLISM
        # Faster & Bigger = more expensive to live
        cost = BASE_ENERGY_LOSS + (self.speed_trait * SPEED_COST) + (self.size_trait * SIZE_COST)
        self.energy -= cost
        self.age += 1
        
        # 3. EAT
        if not self.is_carnivore:
            # Herbivore: Eat Plants
            # Find plants nearby (brute force for simplicity)
            surviving_plants = []
            for p in plants:
                dist = np.hypot(p[0]-self.x, p[1]-self.y)
                if dist < self.size_trait / 2: # Collision
                    self.energy += 20
                else:
                    surviving_plants.append(p)
            return surviving_plants, others # Return modified lists
            
        else:
            # Carnivore: Eat Herbivores
            # Find prey
            surviving_others = []
            eaten_count = 0
            for o in others:
                if o is self: 
                    surviving_others.append(o)
                    continue
                    
                dist = np.hypot(o.x - self.x, o.y - self.y)
                # Kill condition: Must be close AND (larger OR faster)
                # Simplified: Just size overlap
                if dist < self.size_trait/2 and not o.is_carnivore:
                    self.energy += 40 # Meat is high energy
                    eaten_count += 1
                else:
                    surviving_others.append(o)
            return plants, surviving_others

# --- WORLD STATE ---
organisms = [Organism(np.random.rand()*WIDTH, np.random.rand()*HEIGHT) for _ in range(INIT_POPULATION)]
plants = [[np.random.rand()*WIDTH, np.random.rand()*HEIGHT] for _ in range(50)]

# --- VISUALIZATION ---
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2)

# Map View
ax_map = fig.add_subplot(gs[:, 0])
ax_map.set_xlim(0, WIDTH); ax_map.set_ylim(0, HEIGHT)
ax_map.set_title("The Arena (Blue=Prey, Red=Predator)")

# Stats View 1: Population
ax_stats = fig.add_subplot(gs[0, 1])
pop_line_prey, = ax_stats.plot([], [], 'b-', label='Prey')
pop_line_pred, = ax_stats.plot([], [], 'r-', label='Predators')
ax_stats.set_xlim(0, 200); ax_stats.set_ylim(0, 50)
ax_stats.legend(loc='upper right')
ax_stats.set_title("Population Dynamics")

# Stats View 2: Gene Pool
ax_genes = fig.add_subplot(gs[1, 1])
ax_genes.set_title("Average Speed Gene")
ax_genes.set_xlim(0, 200); ax_genes.set_ylim(0, 1)
speed_line, = ax_genes.plot([], [], 'g-', label='Avg Speed')

# Graphics containers
scatter_orgs = ax_map.scatter([], [], s=[], c=[])
scatter_plants = ax_map.scatter([], [], c='green', s=10, marker='.')

history_prey, history_pred, history_speed = [], [], []

def animate(frame):
    global organisms, plants
    
    # 1. Spawn Plants
    for _ in range(PLANT_GROWTH_RATE):
        plants.append([np.random.rand()*WIDTH, np.random.rand()*HEIGHT])
        
    # 2. Update Organisms
    new_orgs = []
    
    # Logic is slightly complex to handle list modification
    # We do a two-pass approach for safety
    
    # Pass 1: Movement & Eating
    # Note: In a real efficient sim, we'd use a spatial grid (QuadTree)
    # Here, for code simplicity, predators eat immediately
    
    current_orgs = organisms[:]
    np.random.shuffle(current_orgs) # Randomize turn order
    
    survivors = []
    
    for org in current_orgs:
        # Check if org was already eaten in this frame (not in survivors list check, but logic check)
        # For simplicity, we assume simultaneous updates or allow "overkill"
        
        p_list, o_list = org.update(plants, current_orgs)
        plants = p_list
        # Note: o_list handling is tricky in simple loops. 
        # Simplified: We just keep the org if it has energy
        
        if org.energy > 0:
            survivors.append(org)
            
            # Reproduction
            if org.energy > 150:
                org.energy /= 2
                offspring = Organism(org.x, org.y, org.dna, org.energy)
                survivors.append(offspring)
    
    # Interaction (Carnivory) - Brute force "Cleanup"
    # We remove prey that are touching carnivores
    final_survivors = []
    killers = [o for o in survivors if o.is_carnivore]
    
    for o in survivors:
        eaten = False
        if not o.is_carnivore:
            for k in killers:
                dist = np.hypot(o.x - k.x, o.y - k.y)
                if dist < k.size_trait/2:
                    k.energy += 30 # Meal reward
                    eaten = True
                    break
        if not eaten:
            final_survivors.append(o)
            
    organisms = final_survivors
    
    # 3. Update Visuals
    # Map
    if organisms:
        offsets = [[o.x, o.y] for o in organisms]
        colors = [o.color for o in organisms]
        sizes = [o.size_trait * 10 for o in organisms]
        scatter_orgs.set_offsets(offsets)
        scatter_orgs.set_facecolors(colors)
        scatter_orgs.set_sizes(sizes)
    else:
        scatter_orgs.set_offsets(np.zeros((0, 2)))

    if plants:
        scatter_plants.set_offsets(plants)
    else:
        scatter_plants.set_offsets(np.zeros((0, 2)))
        
    # Stats
    n_prey = sum(1 for o in organisms if not o.is_carnivore)
    n_pred = sum(1 for o in organisms if o.is_carnivore)
    avg_speed = np.mean([o.dna[0] for o in organisms]) if organisms else 0
    
    history_prey.append(n_prey)
    history_pred.append(n_pred)
    history_speed.append(avg_speed)
    
    # Scroll graph
    if len(history_prey) > 200:
        history_prey.pop(0); history_pred.pop(0); history_speed.pop(0)
        
    x = range(len(history_prey))
    pop_line_prey.set_data(x, history_prey)
    pop_line_pred.set_data(x, history_pred)
    speed_line.set_data(x, history_speed)
    
    return scatter_orgs, scatter_plants, pop_line_prey, pop_line_pred, speed_line

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.show()