from .neuron import IzhikevichNeuron
from .synapse import Synapse

class CorticalNetwork:
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.time = 0.0 # Global network clock (ms)
    
    def add_neuron(self, neuron=None):
        if neuron is None:
            neuron = IzhikevichNeuron()
        self.neurons.append(neuron)
        return neuron

    def connect(self, pre, post, weight=10.0):
        syn = Synapse(pre, post, weight)
        self.synapses.append(syn)
        return syn

    def step(self, external_currents=None, dopamine=0.1):
        """
        Advance the network by 1ms.
        dopamine: Global neuromodulator level (0.0 to 1.0+)
        """
        if external_currents is None:
            external_currents = [0.0] * len(self.neurons)
            
        # 1. Calculate synaptic inputs
        synaptic_inputs = [0.0] * len(self.neurons)
        for syn in self.synapses:
            try:
                post_idx = self.neurons.index(syn.post)
                synaptic_inputs[post_idx] += syn.get_output()
            except ValueError:
                continue

        # 2. Update all neurons
        active_spikes = []
        for i, neuron in enumerate(self.neurons):
            total_current = external_currents[i] + synaptic_inputs[i]
            fired = neuron.step(total_current)
            
            if fired:
                active_spikes.append(i)
                neuron.last_fired_time = self.time

        # 3. Apply Plasticity (Learning)
        # Only learn if dopamine is present (gating)
        if dopamine > 0.0:
            for syn in self.synapses:
                syn.apply_stdp(self.time, dopamine)

        self.time += 1.0
        return active_spikes
