import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE BIOLOGY (MATH) ---
# A Neuron is just a math function: Output = Activation(Sum(Inputs * Weights))

def sigmoid(x):
    # Activation function: Turns numbers into 0-1 signals (like a neuron firing)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Used for learning: "How much did this neuron mess up?"
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):
        # We are building a brain with:
        # 2 Input Neurons (Eyes)
        # 4 Hidden Neurons (Thinking processing layer)
        # 1 Output Neuron (Answer)
        
        # Weights = Synapses (connections between neurons)
        # Initialized randomly, like a baby's brain
        self.weights0 = 2 * np.random.random((2, 4)) - 1 # Synapses Input -> Hidden
        self.weights1 = 2 * np.random.random((4, 1)) - 1 # Synapses Hidden -> Output

    def think(self, inputs):
        # Forward Propagation: Signal flows through the brain
        self.layer1 = sigmoid(np.dot(inputs, self.weights0)) # Hidden layer thoughts
        self.output = sigmoid(np.dot(self.layer1, self.weights1)) # Final decision
        return self.output

    def learn(self, inputs, correct_answers):
        # Backpropagation: The brain reflecting on its mistakes
        
        # 1. Try to think
        self.think(inputs)
        
        # 2. Calculate Error: "How wrong was I?"
        error_layer2 = correct_answers - self.output
        delta_layer2 = error_layer2 * sigmoid_derivative(self.output)
        
        # 3. Calculate Hidden Layer Error: "Which hidden neuron is to blame?"
        error_layer1 = delta_layer2.dot(self.weights1.T)
        delta_layer1 = error_layer1 * sigmoid_derivative(self.layer1)
        
        # 4. Update Weights (Neuroplasticity)
        # Strengthen connections that were right, weaken those that were wrong
        self.weights1 += self.layer1.T.dot(delta_layer2)
        self.weights0 += inputs.T.dot(delta_layer1)

# --- 2. THE SCHOOL (TRAINING) ---

if __name__ == "__main__":
    brain = NeuralNetwork()
    
    # The Logic Puzzle (XOR Gate)
    # This is hard for simple computers because it's not linear.
    # Input A, Input B -> Correct Output
    # 0, 0 -> 0
    # 0, 1 -> 1
    # 1, 0 -> 1
    # 1, 1 -> 0
    training_inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
    training_outputs = np.array([[0,1,1,0]]).T
    
    print("Newborn Brain Random Guessing:")
    print(brain.think(np.array([1, 0])))
    
    # Train the brain 10,000 times (Simulating years of school)
    loss_history = []
    
    print("\nTraining...")
    for i in range(10000):
        brain.learn(training_inputs, training_outputs)
        
        # Record error for graphing
        if i % 100 == 0:
            loss = np.mean(np.square(training_outputs - brain.output))
            loss_history.append(loss)

    print("\nAdult Brain Results (After Learning):")
    print(f"Input [0, 0] -> Brain thinks: {brain.think(np.array([0,0]))[0]:.4f} (Target: 0)")
    print(f"Input [1, 0] -> Brain thinks: {brain.think(np.array([1,0]))[0]:.4f} (Target: 1)")
    print(f"Input [0, 1] -> Brain thinks: {brain.think(np.array([0,1]))[0]:.4f} (Target: 1)")
    print(f"Input [1, 1] -> Brain thinks: {brain.think(np.array([1,1]))[0]:.4f} (Target: 0)")
    
    # --- 3. VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    
    # Graph 1: Learning Curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("The Learning Process")
    plt.xlabel("Study Time (x100)")
    plt.ylabel("Error / Confusion")
    
    # Graph 2: Visualizing the Brain's Decision Map
    plt.subplot(1, 2, 2)
    plt.title("Brain's Logic Map (Heatmap)")
    
    # Generate a grid of points to see what the brain thinks of the whole world
    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
    grid_flat = np.c_[grid_x.ravel(), grid_y.ravel()]
    predictions = brain.think(grid_flat).reshape(grid_x.shape)
    
    plt.imshow(predictions, extent=[-0.5, 1.5, -0.5, 1.5], origin='lower', cmap='bwr', alpha=0.8)
    plt.colorbar(label="Neuron Activation")
    plt.scatter([0,1], [0,1], c='blue', s=100, label="False (0)") # Target 0
    plt.scatter([0,1], [1,0], c='red', s=100, label="True (1)")   # Target 1
    plt.legend()
    
    plt.tight_layout()
    plt.show()