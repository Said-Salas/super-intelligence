import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_SIZE = 64
RESERVOIR_SIZE = 100
TARGET_FIRING_RATE = 0.1  # Homeostasis goal (10% activity)
LEARNING_RATE = 0.01      # Hebbian rate
DECAY_RATE = 0.001        # Weight decay (entropy)
HOMEOSTATIC_RATE = 0.05   # How fast neurons adapt their excitability
STDP_WINDOW = 5           # Time window for causality

class BiologicalBrain:
    def __init__(self):
        # 1. STRUCTURE
        # Synapses: Weights between neurons (Reservoir -> Reservoir)
        # Initialized sparsely (mostly zeros) to allow room for "growth"
        self.weights = np.random.randn(RESERVOIR_SIZE, RESERVOIR_SIZE) * 0.1
        mask = np.random.rand(RESERVOIR_SIZE, RESERVOIR_SIZE) > 0.8
        self.weights *= mask # 80% sparse start
        
        # Input Connections (Sensors -> Brain)
        self.input_weights = np.random.randn(INPUT_SIZE, RESERVOIR_SIZE) * 0.5
        
        # 2. STATE
        self.activity = np.zeros(RESERVOIR_SIZE)      # Current firing state (v)
        self.firing_history = np.zeros((STDP_WINDOW, RESERVOIR_SIZE)) # For STDP
        
        # 3. HOMEOSTASIS (Self-Regulation)
        # Each neuron has a dynamic "threshold" (excitability)
        # If it fires too much, threshold goes up (harder to fire)
        # If it fires too little, threshold goes down (easier to fire)
        self.thresholds = np.zeros(RESERVOIR_SIZE) 

    def step(self, sensory_input):
        """
        One millisecond of brain time.
        1. Process Input
        2. Internal Dynamics
        3. Plasticity (Learning)
        """
        # --- A. DYNAMICS (The "Thinking") ---
        
        # 1. Integrate signals
        # Input contribution + Recurrent contribution
        # v(t) = tanh( Win * u + Wrec * v(t-1) - Thresholds )
        
        external_drive = np.dot(sensory_input, self.input_weights)
        internal_drive = np.dot(self.activity, self.weights)
        
        # Neural Activation Function (Activation Potential)
        potential = external_drive + internal_drive - self.thresholds
        
        # Firing Rule: Non-linear activation (ReLU-like but smooth)
        # If potential > 0, we fire. The stronger the potential, the stronger the rate.
        new_activity = np.maximum(0, np.tanh(potential))
        
        # Update State
        self.activity = new_activity
        
        # Update History (Rolling buffer for STDP)
        self.firing_history = np.roll(self.firing_history, 1, axis=0)
        self.firing_history[0] = self.activity

        # --- B. PLASTICITY (The "Rewiring") ---
        self._plasticity_update()
        
        return self.activity

    def _plasticity_update(self):
        """
        The mechanism of self-organization.
        """
        # 1. HEBBIAN LEARNING (Oja's Rule variant)
        # "Cells that fire together, wire together"
        # dw = rate * (Post * Pre - decay * Post^2 * w)
        # We use a simplified Hebbian: Pre * Post
        
        # Current activity (Post-synaptic)
        post = self.activity.reshape(1, -1) # Row vector
        # Previous activity (Pre-synaptic) - simplified to just last step for efficiency
        pre = self.firing_history[1].reshape(-1, 1) # Column vector
        
        # The Hebbian Term: Correlation matrix
        hebbian_growth = np.dot(pre, post)
        
        # Apply changes to weights (Only where connections exist - functional plasticity)
        # To simulate "structural" plasticity, we allows small weights to grow from zero
        self.weights += LEARNING_RATE * hebbian_growth
        
        # 2. HOMEOSTASIS (The "Thermostat")
        # Regulate firing rates
        current_rate = np.mean(self.firing_history, axis=0)
        error = current_rate - TARGET_FIRING_RATE
        
        # Update Thresholds (Intrensic Plasticity)
        # Too active -> Increase threshold (suppress)
        # Too quiet -> Decrease threshold (sensitize)
        self.thresholds += HOMEOSTATIC_RATE * error
        
        # 3. SYNAPTIC SCALING (Normalization)
        # Prevent runaway explosion of weights
        # If a neuron receives too much total input, scale down ALL its incoming weights
        total_in = np.sum(np.abs(self.weights), axis=0)
        scaling_factor = 1.0 / (total_in + 1e-6) # Inverse
        # Only scale if too high
        overactive_indices = np.where(total_in > 5.0)[0]
        self.weights[:, overactive_indices] *= 0.95 # Soft scaling
        
        # 4. STRUCTURAL PLASTICITY (Life & Death)
        # Synaptogenesis: Randomly spawn weak connections to "bored" neurons
        bored_neurons = np.where(current_rate < 0.01)[0]
        if len(bored_neurons) > 0:
            # Connect random source to bored destination
            src = np.random.randint(0, RESERVOIR_SIZE, len(bored_neurons))
            self.weights[src, bored_neurons] += np.random.rand(len(bored_neurons)) * 0.1
            
        # Pruning: Kill weak connections
        # "Use it or lose it"
        mask = np.abs(self.weights) > 0.05
        self.weights *= mask
        
        # Clamp weights to prevent explosion
        self.weights = np.clip(self.weights, -2.0, 2.0)
        self.weights -= DECAY_RATE * self.weights # Entropy (Passive decay)

# Helper for visualization
def get_complexity_score(brain):
    # A crude approximation of Phi (Integrated Information)
    # Measure the diversity of the weight matrix (Entropy of weights)
    # Higher entropy = more complex structure (not just random, but distributed)
    w = brain.weights.flatten()
    hist, _ = np.histogram(w, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy
