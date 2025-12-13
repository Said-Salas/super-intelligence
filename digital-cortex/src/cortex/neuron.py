import random

class Neuron:
    def __init__(self, neuron_id, threshold=1.0, decay=0.9):
        self.id = neuron_id
        self.potential = 0.0      # Current voltage (membrane potential)
        self.threshold = threshold # Voltage required to fire
        self.decay = decay        # How fast it forgets (leaks voltage)
        self.synapses = []        # Outgoing connections to other neurons
        self.is_refractory = False # Recovering after a spike

    def connect(self, other_neuron, weight=0.1):
        """Connect this neuron to another."""
        self.synapses.append({"neuron": other_neuron, "weight": weight})

    def receive_spike(self, input_current):
        """Receive energy from an upstream neuron or sensor."""
        if not self.is_refractory:
            self.potential += input_current

    def tick(self):
        """One millisecond of time passes."""
        output_signal = 0

        # 1. Leak: The neuron naturally loses charge (forgetting)
        self.potential *= self.decay

        # 2. Fire: Did we hit the threshold?
        if self.potential >= self.threshold:
            output_signal = 1
            self.fire()
            self.potential = 0.0 # Reset
            self.is_refractory = True
        else:
            self.is_refractory = False
        
        return output_signal

    def fire(self):
        """Send spikes to all connected neurons."""
        # print(f"Neuron {self.id} SPIKED!")
        for synapse in self.synapses:
            # Hebbian Learning could happen here: 
            # If the target neuron fires right after us, strengthen this weight.
            synapse["neuron"].receive_spike(synapse["weight"])

class Brain:
    def __init__(self, size=10):
        self.neurons = [Neuron(i) for i in range(size)]
        
        # Create random connections (The "Liquid")
        for n in self.neurons:
            target = random.choice(self.neurons)
            if target != n:
                n.connect(target, weight=0.5)

    def stimulate(self, neuron_idx, strength=1.5):
        """Zap a specific neuron"""
        self.neurons[neuron_idx].receive_spike(strength)

    def run_cycle(self):
        active_neurons = []
        for n in self.neurons:
            if n.tick() == 1:
                active_neurons.append(n.id)
        return active_neurons

if __name__ == "__main__":
    brain = Brain(size=5)
    
    print("--- Stimulation Cycle ---")
    brain.stimulate(0) # Zap Neuron 0
    
    for t in range(10):
        spikes = brain.run_cycle()
        print(f"Time {t}ms: Spikes at {spikes}")
