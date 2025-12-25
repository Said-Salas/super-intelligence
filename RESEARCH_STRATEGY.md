# The Path to Superintelligence: A Research Strategy

## 1. How to THINK: First Principles Thinking
To solve a problem this large, you must abandon analogy ("It's like a brain") and embrace physics/math ("It optimizes free energy").

*   **Inversion:** Instead of asking "How do I create intelligence?", ask "What prevents this rock from being intelligent?" (Lack of state, lack of sensory-motor loop, lack of goal/homeostasis).
*   **Emergence over Engineering:** You cannot hard-code a thought. You can only code the *rules* of the substrate (physics of the neuron) and the *environment*. Intelligence must emerge.
*   **The Map is Not the Territory:** A Python class named `Neuron` is not a neuron. Always ask: "What biological constraint did I ignore for convenience?" (e.g., energy consumption, neurotransmitter diffusion).

## 2. Effective Research: The Feedback Loop
Don't fall into the trap of reading papers for months without writing code.

*   **Read -> Hypothesize -> Simulate -> Fail -> Repeat.**
*   **The "Toy Model" Philosophy:** If you can't demonstrate the principle with 10 neurons, you won't understand it with 10 billion. Your `pipeline.py` is a perfect example of this.
*   **Cross-Pollinate:** 
    *   Read **Neuroscience** to understand *what* to build.
    *   Read **Computer Science** to understand *how* to optimize it.
    *   Read **Control Theory** (Cybernetics) to understand *stability*.

## 3. Where to Find Answers
The answers are rarely in the latest "State of the Art" AI paper. Those optimize current paradigms. You need paradigm shifts.

*   **Biology:** *Principles of Neural Design* (Sterling/Laughlin), *Spikes* (Rieke et al.). Look at how the visual cortex *actually* processes edges.
*   **Cybernetics:** Norbert Wiener, W. Ross Ashby (*Design for a Brain*). They solved the "purpose" problem before computers existed.
*   **Active Inference / Free Energy Principle:** Karl Friston. The math of how living things resist entropy.
*   **Evolutionary Dynamics:** How does complexity increase over time?

## 4. Priorities: What Matters?
1.  **Plasticity (Learning):** If it can't learn from a single event, it's not intelligent. (We started this with STDP).
2.  **Homeostasis (Motivation):** Why does it *want* to learn? It needs internal drives (energy, curiosity) that it must satisfy.
3.  **Efficiency:** The human brain runs on 20 Watts. If your model scales linearly with compute, it's a dead end.
4.  **Embodiment:** Intelligence exists to move a body through a world. A disembodied "brain in a jar" (LLM) is severely limited.

## 5. Safety: Alignment by Architecture
You cannot "patch" safety into a superintelligence later. It must be fundamental.

*   **Shared Substrate:** If the AI functions like a biological entity (homeostasis, pain/pleasure), it can understand human suffering.
*   **Dependence:** Build it such that its "thriving" is coupled with ours.
*   **Gradual Agency:** Start with a worm, then a mouse, then a dog. Don't jump to a god. Verify safety at every level of complexity.

