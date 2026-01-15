import numpy as np

class BiologicalBrain:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # --- SYNAPTIC MATRICES (The Connectome) ---
        # We initialize with sparse random connections to simulate a "young" brain
        # Weights range: -1.0 to 1.0
        self.W_in = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
        self.W_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, hidden_size)) # Recurrent connections
        self.W_out = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
        
        # Mask to enforce sparsity (Structural constraint)
        # 1 = Synapse exists, 0 = No synapse
        self.mask_in = (np.random.rand(input_size, hidden_size) < 0.2).astype(float)
        self.mask_hidden = (np.random.rand(hidden_size, hidden_size) < 0.1).astype(float)
        self.mask_out = (np.random.rand(hidden_size, output_size) < 0.2).astype(float)
        
        # Apply masks initially
        self.W_in *= self.mask_in
        self.W_hidden *= self.mask_hidden
        self.W_out *= self.mask_out
        
        # --- NEURON STATES ---
        self.state = np.zeros(hidden_size)
        self.firing_history = np.zeros(hidden_size) # For homeostasis (moving average)
        
        # --- PLASTICITY PARAMETERS ---
        self.learning_rate = 0.01
        self.target_firing_rate = 0.1 # We want sparse activity (10% of neurons firing)
        self.homeostasis_rate = 0.05  # How fast neurons adapt their sensitivity
        self.pruning_threshold = 0.05 # Synapses weaker than this die
        self.growth_chance = 0.01     # Probability of a new synapse forming per step

    def activation(self, x):
        # Tanh gives -1 to 1, representing inhibition/excitation
        return np.tanh(x)

    def forward(self, inputs):
        """
        Process one time step.
        inputs: array of shape (input_size,)
        """
        # 1. Integrate signals
        # Input contribution
        input_signal = np.dot(inputs, self.W_in)
        
        # Internal recurrence (Memory/Context)
        internal_signal = np.dot(self.state, self.W_hidden)
        
        # Total potential
        potential = input_signal + internal_signal
        
        # 2. Fire
        self.state = self.activation(potential)
        
        # 3. Output
        output = self.activation(np.dot(self.state, self.W_out))
        
        return output

    def plasticity_step(self, inputs):
        """
        The 'Sleep' or 'consolidation' phase where wiring updates happen.
        """
        # --- 1. HEBBIAN LEARNING (STDP simplified) ---
        # "Cells that fire together, wire together"
        # Increase weight if Input > 0 AND Output > 0
        
        # Input -> Hidden
        # Outer product: inputs[i] * state[j]
        # We only update EXISTING synapses (mask == 1)
        delta_in = np.outer(inputs, self.state) * self.learning_rate
        self.W_in += delta_in * self.mask_in
        
        # Hidden -> Hidden (Recurrent)
        delta_hidden = np.outer(self.state, self.state) * self.learning_rate
        self.W_hidden += delta_hidden * self.mask_hidden
        
        # --- 2. HOMEOSTASIS (Scaling) ---
        # Update firing history (Exponential Moving Average)
        current_activity = np.abs(self.state)
        self.firing_history = (1 - self.homeostasis_rate) * self.firing_history + \
                              self.homeostasis_rate * current_activity
        
        # Calculate Homeostatic Scaling Factor
        # If history > target, we need to cool down (factor < 0)
        # If history < target, we need to heat up (factor > 0)
        scaling_factor = self.target_firing_rate - self.firing_history
        
        # Apply scaling to ALL incoming weights for each neuron
        # This effectively changes the neuron's "threshold"
        # We broadcast the scaling factor across columns (neurons)
        self.W_in += self.W_in * scaling_factor * 0.1 
        self.W_hidden += self.W_hidden * scaling_factor * 0.1
        
        # --- 3. STRUCTURAL PLASTICITY (Rewiring) ---
        self._structural_changes()
        
        # Clamp weights to prevent explosion
        self.W_in = np.clip(self.W_in, -2.0, 2.0)
        self.W_hidden = np.clip(self.W_hidden, -2.0, 2.0)
        
    def _structural_changes(self):
        # A. PRUNING (Apoptosis of connections)
        # Identify weak synapses
        weak_in = np.abs(self.W_in) < self.pruning_threshold
        weak_hidden = np.abs(self.W_hidden) < self.pruning_threshold
        
        # Remove them (Update mask and weights)
        self.mask_in[weak_in] = 0
        self.W_in[weak_in] = 0
        
        self.mask_hidden[weak_hidden] = 0
        self.W_hidden[weak_hidden] = 0
        
        # B. SYNAPTOGENESIS (Growth)
        # Randomly form new connections to explore new pathways
        # We only add connections where mask is currently 0
        
        # Input -> Hidden
        potential_synapses = np.where(self.mask_in == 0)
        if len(potential_synapses[0]) > 0:
            # Randomly select indices to grow
            num_grow = int(self.input_size * self.hidden_size * self.growth_chance)
            indices = np.random.choice(len(potential_synapses[0]), num_grow)
            
            rows = potential_synapses[0][indices]
            cols = potential_synapses[1][indices]
            
            self.mask_in[rows, cols] = 1
            # Initialize with small random weight
            self.W_in[rows, cols] = np.random.uniform(-0.1, 0.1, num_grow)
            
        # Hidden -> Hidden
        potential_synapses_h = np.where(self.mask_hidden == 0)
        if len(potential_synapses_h[0]) > 0:
            num_grow_h = int(self.hidden_size * self.hidden_size * self.growth_chance)
            indices_h = np.random.choice(len(potential_synapses_h[0]), num_grow_h)
            
            rows_h = potential_synapses_h[0][indices_h]
            cols_h = potential_synapses_h[1][indices_h]
            
            self.mask_hidden[rows_h, cols_h] = 1
            self.W_hidden[rows_h, cols_h] = np.random.uniform(-0.1, 0.1, num_grow_h)

