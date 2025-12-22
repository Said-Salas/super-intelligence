import time
import random
from .brain_model import LiquidBrain, SpikeEncoder

def run_pipeline():
    # 1. Setup Data
    # A simple sequence pattern: A -> B -> C -> A -> B -> C
    text_stream = "ABC" * 100 
    encoder = SpikeEncoder("ABC")
    
    # 2. Setup Brain
    # 3 Inputs (A,B,C), 20 Reservoir Neurons, 3 Output Predictions (A,B,C)
    brain = LiquidBrain(n_inputs=3, n_reservoir=20, n_outputs=3)
    net = brain.net
    
    print("--- STARTING BIOLOGICAL TRAINING PIPELINE ---")
    print("Task: Predict the NEXT character in sequence 'ABCABC...'")
    
    correct_predictions = 0
    total_predictions = 0
    
    # 3. Training Loop
    # We process the stream one character at a time.
    # Each character is presented for 50ms (window).
    
    window_ms = 50
    
    for i in range(len(text_stream) - 1):
        current_char = text_stream[i]
        target_char = text_stream[i+1] # The ground truth next char
        
        # Target Index for checking prediction
        target_idx = encoder.char_to_idx[target_char]
        
        # Reset output spike counters for this window
        output_spikes = [0] * 3 
        
        # --- PRESENT STIMULUS (50ms) ---
        for t in range(window_ms):
            dopamine = 0.1 # Baseline curiosity
            
            # Step
            fired_indices = brain.step(current_char, encoder, dopamine=dopamine)
            
            # Check Output Spikes
            out_start = len(brain.input_neurons) + len(brain.reservoir_neurons)
            
            for fired_idx in fired_indices:
                if fired_idx >= out_start:
                    out_channel = fired_idx - out_start
                    output_spikes[out_channel] += 1
                    
                    # REWARD LEARNING
                    if out_channel == target_idx:
                        # Correct! Massive Dopamine Rush.
                        # We trigger an extra "learning step" with high dopamine to lock it in.
                        brain.net.step(dopamine=5.0) 
                    else:
                        # Wrong! Punishment.
                        brain.net.step(dopamine=-1.0)

        # --- EVALUATE WINDOW ---
        # Who fired the most?
        if sum(output_spikes) > 0:
            pred_idx = output_spikes.index(max(output_spikes))
            pred_char = encoder.idx_to_char[pred_idx]
        else:
            pred_char = "_"
            
        is_correct = (pred_char == target_char)
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
            
        total_predictions += 1
        acc = (correct_predictions / total_predictions) * 100
        
        print(f"In: {current_char} | Pred: {pred_char} (Target: {target_char}) | {status} | Acc: {acc:.1f}%")
        
        # SLEEP/REST (Inter-stimulus interval)
        for _ in range(20):
            brain.net.step() # No input, just physics

if __name__ == "__main__":
    run_pipeline()
