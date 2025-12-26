import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
GRID_SIZE = 100
DIFFUSION_RATE = 0.1    # How fast signals spread
DECAY_RATE = 0.02       # How fast signals fade
REPRODUCTION_THRESHOLD = 0.5  # Need this much chemical signal to grow
OVERCROWDING_LIMIT = 3.0      # Too much signal = stop growing (prevents solid blocks)

class ChemicalWorld:
    def __init__(self):
        # 0 = Empty, 1 = Bacteria
        self.cells = np.zeros((GRID_SIZE, GRID_SIZE))
        # Continuous value grid for chemical signals
        self.chemicals = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Start with one seed in the center
        self.cells[GRID_SIZE//2, GRID_SIZE//2] = 1
        
    def update(self):
        # 1. Bacteria Secrete Chemicals
        # Every living cell adds +1.0 to its location in the chemical grid
        self.chemicals += self.cells * 1.0
        
        # 2. Diffusion (Physics of fluids)
        # Chemicals spread to neighbors: Up, Down, Left, Right
        # Using a simple Laplacian convolution manually for clarity
        padded = np.pad(self.chemicals, 1, mode='wrap')
        # Neighbor sum
        neighbors = (
            padded[0:-2, 1:-1] + # Top
            padded[2:, 1:-1] +   # Bottom
            padded[1:-1, 0:-2] + # Left
            padded[1:-1, 2:]     # Right
        )
        # Diffusion formula: New = Old + Rate * (Average_Neighbors - Old)
        self.chemicals += DIFFUSION_RATE * (neighbors - 4 * self.chemicals)
        
        # 3. Decay (Entropy)
        self.chemicals *= (1 - DECAY_RATE)
        
        # 4. Life & Death Rules (The Logic)
        # Copy grid to avoid updating while reading
        new_cells = self.cells.copy()
        
        # Get indices of all empty spots and filled spots
        # Optimization: We only check relevant areas (edges of the colony)
        # For simplicity, we scan the whole grid (fast enough for 100x100)
        
        rows, cols = np.where(self.cells == 1)
        for r, c in zip(rows, cols):
            # Check neighbors for potential growth
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = (r+dr)%GRID_SIZE, (c+dc)%GRID_SIZE
                
                if self.cells[nr, nc] == 0:
                    # If empty, check chemical level
                    signal = self.chemicals[nr, nc]
                    
                    # GROWTH RULE:
                    # Grow if signal is strong enough (support)
                    # BUT NOT if signal is too high (overcrowding/choking)
                    if signal > REPRODUCTION_THRESHOLD and signal < OVERCROWDING_LIMIT:
                        # Small chance to actually grow (adds randomness/organic look)
                        if np.random.random() < 0.1:
                            new_cells[nr, nc] = 1

        self.cells = new_cells

# --- VISUALIZATION ---
world = ChemicalWorld()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Biofilm Growth: Growth Guided by Chemical Signals")

# Plot 1: The Bacteria (Living Cells)
img_cells = ax1.imshow(world.cells, cmap='Greys', interpolation='nearest')
ax1.set_title("Bacteria Colony")
ax1.axis('off')

# Plot 2: The Chemical Field (Invisible Communication)
img_chem = ax2.imshow(world.chemicals, cmap='inferno', interpolation='bicubic')
ax2.set_title("Chemical Signal Field")
ax2.axis('off')

def animate(frame):
    for _ in range(5): # Speed up: run 5 simulation steps per frame
        world.update()
    
    img_cells.set_data(world.cells)
    img_chem.set_data(world.chemicals)
    # Auto-scale chemical color for better visibility
    img_chem.set_clim(vmin=0, vmax=np.max(world.chemicals))
    
    return img_cells, img_chem

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.show()