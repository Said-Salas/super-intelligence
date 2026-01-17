import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from neuroplasticity import BiologicalBrain, INPUT_SIZE, RESERVOIR_SIZE

# --- THE MATRIX (Sensory Input Generator) ---
def get_sensory_pattern(t):
    # A shifting pattern to force the brain to adapt
    # e.g., A moving "bar" of activity across the input sensors
    pattern = np.zeros(INPUT_SIZE)
    
    # Phase 1: Moving Sine Wave
    phase = t * 0.1
    for i in range(INPUT_SIZE):
        val = np.sin(phase + i * 0.2)
        pattern[i] = max(0, val) # Rectified
        
    # Phase 2: Occasional "Shock" (Flash)
    if t % 200 > 180:
        pattern[:] = 1.0
        
    return pattern

# --- INITIALIZATION ---
brain = BiologicalBrain()

# Setup Visualization
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# 1. Sensory Input View
ax_input = fig.add_subplot(gs[0, 0])
ax_input.set_title("Sensory Input (Retina)")
ax_input.set_ylim(0, 1.2)
line_input, = ax_input.plot([], [], 'g-')

# 2. Brain Activity View (Firing Rates)
ax_activity = fig.add_subplot(gs[0, 1])
ax_activity.set_title("Neural Activity (Firing)")
img_activity = ax_activity.imshow(np.zeros((10, 10)), cmap='inferno', vmin=0, vmax=1)

# 3. Brain Structure View (Connectivity Matrix)
ax_weights = fig.add_subplot(gs[0, 2])
ax_weights.set_title("Synaptic Web (Weights)")
img_weights = ax_weights.imshow(brain.weights, cmap='viridis', vmin=-1, vmax=1)

# 4. Complexity Graph
ax_complexity = fig.add_subplot(gs[1, :])
ax_complexity.set_title("Neural Complexity (Differentiation)")
ax_complexity.set_xlim(0, 500)
line_complexity, = ax_complexity.plot([], [], 'b-', label='Complexity')
history_complexity = []

# --- SIMULATION LOOP ---
def animate(frame):
    # 1. Generate Reality
    inputs = get_sensory_pattern(frame)
    
    # 2. Brain Step
    # Run 5ms of brain time per frame for speed
    for _ in range(5):
        brain.step(inputs)
    
    # 3. Visualization Updates
    
    # Input
    line_input.set_data(range(INPUT_SIZE), inputs)
    ax_input.set_xlim(0, INPUT_SIZE)
    
    # Activity (Reshaped to 10x10 for grid view)
    img_activity.set_data(brain.activity.reshape(10, 10))
    
    # Weights (The Physical Rewiring!)
    img_weights.set_data(brain.weights)
    
    # Complexity Metric
    # Check "how structured" the brain is becoming
    from neuroplasticity import get_complexity_score
    score = get_complexity_score(brain)
    history_complexity.append(score)
    
    # Scroll graph
    if len(history_complexity) > 500:
        history_complexity.pop(0)
    
    line_complexity.set_data(range(len(history_complexity)), history_complexity)
    ax_complexity.set_ylim(min(history_complexity)-0.1, max(history_complexity)+0.1)
    
    return line_input, img_activity, img_weights, line_complexity

# Start Simulation
print("Entering The Womb...")
ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)
plt.tight_layout()
plt.show()
