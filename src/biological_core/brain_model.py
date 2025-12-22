import random
from .network import CorticalNetwork
from .neuron import IzhikevichNeuron

class SpikeEncoder:
    def __init__(self, chars="ABC"):
        self.vocab = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.n_channels = len(chars)

    def encode(self, char):
        """Returns input currents for a 1ms step."""
        currents = [0.0] * self.n_channels
        if char in self.char_to_idx:
            idx = self.char_to_idx[char]
            currents[idx] = 40.0 # Strong stimulation for this channel
        return currents

class LiquidBrain:
    def __init__(self, n_inputs, n_reservoir, n_outputs):
        self.net = CorticalNetwork()
        self.input_neurons = []
        self.reservoir_neurons = []
        self.output_neurons = []
        
        # 1. Create Layers
        for _ in range(n_inputs):
            self.input_neurons.append(self.net.add_neuron())
            
        for _ in range(n_reservoir):
            n = self.net.add_neuron()
            # Randomize reservoir neuron parameters for diversity
            n.params = (0.02, 0.2, -65.0 + random.uniform(-5,5), 8.0 + random.uniform(-2,2))
            self.reservoir_neurons.append(n)
            
        for _ in range(n_outputs):
            self.output_neurons.append(self.net.add_neuron())

        # 2. Wiring: Input -> Reservoir (Sparse)
        for inp in self.input_neurons:
            # Connect to random 30% of reservoir
            for res in self.reservoir_neurons:
                if random.random() < 0.3:
                    self.net.connect(inp, res, weight=random.uniform(5, 15))

        # 3. Wiring: Reservoir -> Reservoir (Recurrent, Sparse)
        # This creates the "Liquid" memory
        for n1 in self.reservoir_neurons:
            for n2 in self.reservoir_neurons:
                if n1 == n2: continue
                if random.random() < 0.1: # 10% connectivity
                    w = random.uniform(-5, 10) # Mixed Excitatory/Inhibitory
                    self.net.connect(n1, n2, weight=w)

        # 4. Wiring: Reservoir -> Output (Dense, Plastic)
        # These are the weights we want to LEARN via STDP + Dopamine
        for res in self.reservoir_neurons:
            for out in self.output_neurons:
                syn = self.net.connect(res, out, weight=random.uniform(0, 5))
                syn.lr = 0.5 # High learning rate for readout
    
    def step(self, char_input, encoder, dopamine=0.1):
        # Map char to currents for input layer
        input_currents = encoder.encode(char_input)
        
        # Pad with 0s for reservoir and output neurons
        total_currents = input_currents + \
                        [0.0] * len(self.reservoir_neurons) + \
                        [0.0] * len(self.output_neurons)
        
        return self.net.step(total_currents, dopamine=dopamine)
