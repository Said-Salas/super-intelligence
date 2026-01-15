import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
GRID_SIZE = 10
# Actions: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.1  # Chance to move randomly (to discover new paths)

class SmartBacterium:
    def __init__(self):
        self.pos = [GRID_SIZE//2, GRID_SIZE//2] # Start in middle
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4)) # The "Brain"
        self.score = 0
        
    def choose_action(self):
        # Epsilon-greedy: sometimes explore, usually exploit learned knowledge
        if np.random.uniform(0, 1) < EXPLORATION_RATE:
            return np.random.randint(4) # Random move
        else:
            # Choose best known move for current position
            r, c = self.pos
            return np.argmax(self.q_table[r, c])

    def step(self, action, food_map):
        # Calculate new position
        dr, dc = ACTIONS[action]
        new_r = max(0, min(GRID_SIZE-1, self.pos[0] + dr))
        new_c = max(0, min(GRID_SIZE-1, self.pos[1] + dc))
        
        # Calculate Reward
        reward = -1 # Cost of living/moving
        if food_map[new_r, new_c] == 1:
            reward = 50 # Yummy!
            self.score += 1
            food_map[new_r, new_c] = 0 # Eat the food
        
        # Q-Learning: Update the brain
        # Q(s,a) = Q(s,a) + lr * (reward + discount * max(Q(next_s)) - Q(s,a))
        old_q = self.q_table[self.pos[0], self.pos[1], action]
        max_future_q = np.max(self.q_table[new_r, new_c])
        new_q = old_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - old_q)
        self.q_table[self.pos[0], self.pos[1], action] = new_q
        
        # Move
        self.pos = [new_r, new_c]
        return food_map

# --- VISUALIZATION SETUP ---
fig, ax = plt.subplots(figsize=(6, 6))
bacterium = SmartBacterium()
food_map = np.zeros((GRID_SIZE, GRID_SIZE))

# Place permanent food sources (replenishing)
food_locations = [(1, 1), (1, 8), (8, 1), (8, 8)]

def update(frame):
    global food_map
    
    # 1. Replenish food if empty (so it keeps learning)
    for fr, fc in food_locations:
        if food_map[fr, fc] == 0 and np.random.random() < 0.05: # Random respawn
             food_map[fr, fc] = 1
             
    # 2. Bacteria thinks and moves
    action = bacterium.choose_action()
    food_map = bacterium.step(action, food_map)
    
    # 3. Draw
    ax.clear()
    # Draw Grid
    ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1)); ax.set_yticks(np.arange(-.5, GRID_SIZE, 1))
    ax.grid(color='black', linestyle='-', linewidth=1)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Draw Food
    rows, cols = np.where(food_map == 1)
    ax.scatter(cols, rows, c='green', s=200, label='Food', marker='s')
    
    # Draw Bacteria
    ax.scatter(bacterium.pos[1], bacterium.pos[0], c='red', s=300, label='Bacteria')
    
    # Draw Brain (Visualize the Q-values as arrows)
    # This shows what the bacteria "wants" to do in each cell
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            best_move = np.argmax(bacterium.q_table[r, c])
            max_val = np.max(bacterium.q_table[r, c])
            if max_val > 0.1: # Only draw if it has learned something here
                dy, dx = ACTIONS[best_move]
                ax.arrow(c, r, dx*0.3, dy*0.3, head_width=0.2, color='blue', alpha=0.3)

    ax.set_title(f"Smart Bacteria (Q-Learning)\nScore: {bacterium.score}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()