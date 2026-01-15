import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neuroplasticity import BiologicalBrain

# --- CONFIGURATION ---
INPUT_SIZE = 10
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
STEPS = 1000

# Initialize Brain
brain = BiologicalBrain(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

# --- SENSORY INPUT GENERATOR ---
# We create a repeating pattern: 1 -> 2 -> 3 -> ... -> 10 -> 1 ...
# This tests if the brain can learn the "sequence" structure
def get_input_pattern(t):
    pattern = np.zeros(INPUT_SIZE)
    # Moving activation
    idx = int(t / 5) % INPUT_SIZE 
    pattern[idx] = 1.0
    # Add some noise
    pattern += np.random.normal(0, 0.1, INPUT_SIZE)
    return pattern

# --- VISUALIZATION SETUP ---
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2)

# 1. Connectome (The Wiring)
ax_weights = fig.add_subplot(gs[0, 0])
img_weights = ax_weights.imshow(brain.W_hidden, cmap='viridis', vmin=-1, vmax=1)
ax_weights.set_title("Recurrent Synapses (The Connectome)")
plt.colorbar(img_weights, ax=ax_weights)

# 2. Activity (The Firing)
ax_activity = fig.add_subplot(gs[0, 1])
bar_activity = ax_activity.bar(range(HIDDEN_SIZE), brain.state)
ax_activity.set_ylim(-1, 1)
ax_activity.set_title("Neural Activity (Snapshot)")

# 3. Connectivity Stats (Evolution)
ax_stats = fig.add_subplot(gs[1, :])
history_synapses = []
history_activity = []
line_synapses, = ax_stats.plot([], [], 'b-', label='Synapse Count')
ax_stats.set_xlim(0, STEPS)
ax_stats.set_title("Structural Evolution")
ax_stats.set_ylabel("Total Synapses")
ax_stats.legend()

# For twin axis (avg activity)
ax_stats2 = ax_stats.twinx()
line_activity, = ax_stats2.plot([], [], 'r-', label='Avg Firing Rate', alpha=0.5)
ax_stats2.set_ylabel("Avg Activity", color='r')
ax_stats2.set_ylim(0, 1.0)

def animate(frame):
    # 1. Run Simulation Step
    inputs = get_input_pattern(frame)
    brain.forward(inputs)
    brain.plasticity_step(inputs)
    
    # 2. Update Visuals
    # Connectome
    img_weights.set_data(brain.W_hidden * brain.mask_hidden)
    
    # Activity
    for rect, h in zip(bar_activity, brain.state):
        rect.set_height(h)
        
    # Stats
    num_synapses = np.sum(brain.mask_hidden)
    avg_activity = np.mean(np.abs(brain.state))
    
    history_synapses.append(num_synapses)
    history_activity.append(avg_activity)
    
    # Scroll if history is too long (optional, but we set fixed xlim)
    if len(history_synapses) > STEPS:
        # Just creating a rolling window effect would require updating xlim
        pass
        
    line_synapses.set_data(range(len(history_synapses)), history_synapses)
    line_activity.set_data(range(len(history_activity)), history_activity)
    
    # Dynamic scaling for synapses graph
    if len(history_synapses) > 1:
        ax_stats.set_ylim(min(history_synapses)*0.9, max(history_synapses)*1.1)
    
    return img_weights, line_synapses, line_activity, *bar_activity

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=50, blit=False)
plt.tight_layout()
plt.show()

