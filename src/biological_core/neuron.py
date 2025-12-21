import math
import random

class IzhikevichNeuron:
    def __init__(self, a=0.02, b=0.2, c=-65.0, d=8.0):
        # The DNA (Parameters)
        self.params = (a, b, c, d)
        
        # The State
        self.v = -65.0  # Membrane Potential (Voltage)
        self.u = b * self.v  # Recovery Variable (Chemistry)

    def step(self, input_current):
        """
        Run the neuron physics for 1 millisecond.
        Returns: True if it spiked, False otherwise.
        """
        v = self.v
        u = self.u
        a, b, c, d = self.params

        # The Differential Equations (Numerically integrated)
        # We perform two 0.5ms steps for stability
        for _ in range(2):
            v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + input_current)
        
        # Update recovery variable (slower dynamics)
        u = u + a * (b * v - u)

        # Check for Spike
        fired = False
        if v >= 30.0:
            # Action Potential Reached!
            fired = True
            v = c      # Reset voltage
            u = u + d  # Chemical aftershock

        # Update state
        self.v = v
        self.u = u
        
        return fired
