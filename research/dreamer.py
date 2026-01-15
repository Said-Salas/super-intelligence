import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE ENVIRONMENT (The Reality) ---
# A complex signal: Sine wave + Noise + Trend
def get_reality_stream(t):
    return np.sin(t * 0.1) + np.sin(t * 0.03) * 0.5 # Two overlapping waves

# --- 2. THE GENERAL INTELLIGENCE (The Predictor) ---
class PredictiveBrain:
    def __init__(self):
        # A simple Recurrent Neural Network (Echo State Network / Reservoir concept)
        # Instead of training ALL weights (slow), we use a chaotic reservoir (fast adaptation)
        
        self.input_size = 1
        self.reservoir_size = 50 # 50 Neurons in the "Cortex"
        self.output_size = 1
        
        # Fixed random connections in the reservoir (The "Nature" part)
        self.W_in = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.input_size))
        self.W_res = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size))
        
        # Stabilize the reservoir (Spectral radius < 1)
        rho = max(abs(np.linalg.eigvals(self.W_res)))
        self.W_res *= 0.95 / rho
        
        # Learnable weights (The "Nurture" part) - Output layer
        self.W_out = np.zeros((self.output_size, self.reservoir_size))
        
        # Memory state
        self.state = np.zeros((self.reservoir_size, 1))
        
        # Online Learning Parameters (Recursive Least Squares - fast adaptation)
        self.P = np.eye(self.reservoir_size) * 1000 # Inverse correlation matrix
        self.forget_factor = 0.99 # Slowly forget old data (allows adapting to new rules)

    def think_and_learn(self, current_sensory_input):
        # 1. PREDICT (Forward Pass)
        # Update internal state based on input + previous state
        # state(t) = tanh( Win*input + Wres*state(t-1) )
        u = np.array([[current_sensory_input]])
        self.state = np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, self.state))
        
        # Generate Prediction for the NEXT moment (t+1)
        prediction = np.dot(self.W_out, self.state)
        
        return prediction[0,0]

    def update_weights(self, actual_next_value):
        # 2. OBSERVE & LEARN (Backward Pass)
        # We learned what actually happened. Update W_out to minimize error.
        # Using RLS (Recursive Least Squares) for one-shot learning
        
        target = np.array([[actual_next_value]])
        
        # Calculate Error (Prediction Error)
        # We don't need to explicitly calculate it for RLS, but conceptually:
        # Error = Target - Prediction
        
        # RLS Math (The "General Learning" Algorithm)
        # k = P * state / (forget + state.T * P * state)
        k = np.dot(self.P, self.state) / (self.forget_factor + np.dot(self.state.T, np.dot(self.P, self.state)))
        
        # W_out = W_out + k * (Target - W_out * state)
        prediction = np.dot(self.W_out, self.state)
        self.W_out += np.dot(k, (target - prediction).T).T
        
        # P = (P - k * state.T * P) / forget
        self.P = (self.P - np.dot(k, np.dot(self.state.T, self.P))) / self.forget_factor

# --- 3. THE EXPERIMENT ---
brain = PredictiveBrain()

history_reality = []
history_prediction = []
history_error = []

# Time steps
time_steps = 500

# Initial "dream" (guess)
last_prediction = 0 

print("System coming online...")

for t in range(time_steps):
    # 1. Reality happens
    current_reality = get_reality_stream(t)
    history_reality.append(current_reality)
    
    # 2. Brain makes a prediction for NOW based on PREVIOUS info
    # (Visualizing what the brain *thought* would happen vs what *did* happen)
    history_prediction.append(last_prediction)
    
    # Calculate Surprise (Free Energy)
    error = abs(current_reality - last_prediction)
    history_error.append(error)
    
    # 3. Brain processes NOW to predict TOMORROW
    # It takes current reality, updates its internal state, and learns
    # Note: We teach it to predict t+1 based on t
    next_reality = get_reality_stream(t + 1)
    
    # The crucial step: The brain updates its weights to better map State(t) -> Reality(t+1)
    brain.update_weights(next_reality)
    
    # The brain generates its expectation for the next moment
    last_prediction = brain.think_and_learn(current_reality)

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Reality vs Dream
ax1.plot(history_reality, 'k-', label='Reality (Sensory Input)', alpha=0.6)
ax1.plot(history_prediction, 'r--', label='Brain Prediction', linewidth=1.5)
ax1.set_title("General Intelligence: Predictive Coding")
ax1.set_ylabel("Signal Value")
ax1.legend()

# Plot 2: Surprise (Error)
ax2.plot(history_error, 'b-', label='Surprise (Prediction Error)')
ax2.set_title("Minimizing Free Energy (Learning)")
ax2.set_ylabel("Error Magnitude")
ax2.set_xlabel("Time (Experience)")
ax2.fill_between(range(time_steps), history_error, color='blue', alpha=0.1)
ax2.legend()

plt.tight_layout()
plt.show()