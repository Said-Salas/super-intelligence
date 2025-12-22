import time
import random
from neuron import IzhikevichNeuron
from network import CorticalNetwork

def run_simulation():
    net = CorticalNetwork()
    
    # 1. Create a simple circuit: Input -> Interneuron -> Output
    n_input = net.add_neuron()  # Index 0
    n_inter = net.add_neuron()  # Index 1
    n_output = net.add_neuron() # Index 2
    
    # 2. Connect them
    # Strong connection from Input to Interneuron
    net.connect(n_input, n_inter, weight=20.0)
    # Weak connection from Interneuron to Output
    net.connect(n_inter, n_output, weight=15.0)
    
    print("--- STARTING CORTICAL SIMULATION ---")
    print("Structure: [0] -> (20.0) -> [1] -> (15.0) -> [2]")
    
    # 3. Run Simulation Loop
    for t in range(100):
        # Inject current ONLY into the first neuron
        # We give it a kick every 10ms
        stim = 20.0 if t % 10 == 0 else 0.0
        inputs = [stim, 0.0, 0.0]
        
        spikes = net.step(inputs)
        
        # Visualization
        status = ["."]*3
        for s in spikes:
            status[s] = "⚡️" # Spike!
            
        print(f"T={t}ms | In: {inputs[0]} | {status[0]} -- {status[1]} -- {status[2]}")
        time.sleep(0.05)

if __name__ == "__main__":
    run_simulation()
