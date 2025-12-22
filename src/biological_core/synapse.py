import math

class Synapse:
    def __init__(self, pre_neuron, post_neuron, weight=10.0, learning_rate=0.1):
        """
        Connects two neurons.
        weight: Strength of the connection (positive=excitatory, negative=inhibitory)
        """
        self.pre = pre_neuron
        self.post = post_neuron
        self.weight = weight
        self.max_weight = 40.0
        self.min_weight = 0.0
        self.lr = learning_rate
        
        # STDP Parameters
        self.tau = 20.0  # Time constant (ms)
        self.A_plus = 0.8 # Potentiation strength
        self.A_minus = 0.85 # Depression strength (usually slightly stronger to stabilize)

    def get_output(self):
        """
        If the pre-synaptic neuron fired in the last step, 
        transmit the weight as current to the post-synaptic neuron.
        """
        if getattr(self.pre, 'fired', False):
            return self.weight
        return 0.0

    def apply_stdp(self, current_time, dopamine=1.0):
        """
        Spike-Timing-Dependent Plasticity (STDP) modulated by Dopamine.
        """
        
        delta_w = 0.0
        
        # 1. Long-Term Potentiation (LTP): Pre -> Post (Causal)
        if getattr(self.post, 'fired', False):
            dt = current_time - self.pre.last_fired_time
            if dt > 0 and dt < 50: # Window of 50ms
                delta_w += self.A_plus * math.exp(-dt / self.tau)

        # 2. Long-Term Depression (LTD): Post -> Pre (Acausal)
        if getattr(self.pre, 'fired', False):
            dt = current_time - self.post.last_fired_time
            if dt > 0 and dt < 50:
                delta_w -= self.A_minus * math.exp(-dt / self.tau)

        # 3. Apply changes modulated by dopamine
        # If dopamine is 0, no learning happens (gating).
        # If dopamine is high, learning is accelerated.
        if delta_w != 0.0:
            self.weight += delta_w * self.lr * dopamine
            self.weight = max(self.min_weight, min(self.max_weight, self.weight))
