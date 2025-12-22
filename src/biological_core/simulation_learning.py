import time
import random
from neuron import IzhikevichNeuron
from network import CorticalNetwork

def run_learning_simulation():
    net = CorticalNetwork()
    
    # 1. Two Neurons: The Teacher (A) and the Student (B)
    n_pre = net.add_neuron()  # Neuron A
    n_post = net.add_neuron() # Neuron B
    
    # 2. Connect them with a weak weight initially
    synapse = net.connect(n_pre, n_post, weight=5.0) # Weak! Won't cause a spike alone.
    
    print("--- STARTING HEBBIAN LEARNING SIMULATION ---")
    print("Goal: Teach Neuron B to fire when Neuron A fires (Association).")
    print(f"Initial Weight: {synapse.weight}")
    
    # 3. Training Protocol
    # We will force them to fire together (A then B) repeatedly.
    # This represents "seeing" an object (A) and hearing its name (B) at the same time.
    
    for t in range(500):
        input_a = 0.0
        input_b = 0.0
        
        # Every 50ms, stimulate A
        if t % 50 == 0:
            input_a = 20.0
            
        # Every 50ms (simultaneously or slightly after), stimulate B
        # To induce LTP, B must fire *after* A. 
        # Since A fires at t%50==0, we stimulate B at t%50==1 to ensure causal link.
        if t % 50 == 2: 
            input_b = 20.0
            
        inputs = [input_a, input_b]
        spikes = net.step(inputs)
        
        if t % 50 == 0:
            # Print snapshot every cycle
            print(f"Time: {t}ms | Weight: {synapse.weight:.2f} | Last Spikes: A={n_pre.last_fired_time}, B={n_post.last_fired_time}")

    print(f"\nFINAL Weight: {synapse.weight:.2f}")
    
    # 4. Test Phase
    print("\n--- TEST PHASE ---")
    print("Stimulating ONLY Neuron A. Will B fire now?")
    
    fired_count = 0
    for t in range(100):
        # Stimulate A only
        input_a = 20.0 if t == 10 else 0.0
        
        spikes = net.step([input_a, 0.0])
        if 1 in spikes: # Did Neuron B (index 1) fire?
            print(f"SUCCESS! Neuron B fired at t={t} (caused by A)!")
            fired_count += 1
            
    if fired_count == 0:
        print("Fail. Weight not strong enough yet.")

if __name__ == "__main__":
    run_learning_simulation()

