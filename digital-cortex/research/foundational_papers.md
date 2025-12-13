# Foundational Research for Digital Cortex

## Core Philosophy: The Brain as a Prediction Engine

We reject the "Feedforward" model of standard Deep Learning. We adopt the **Free Energy Principle** (Friston) and **Predictive Coding**.

### 1. Dendritic Predictive Coding (Mikulasch et al., 2022)
**Source:** [arXiv:2205.05303](https://arxiv.org/abs/2205.05303)

*   **Insight:** Neurons are not point-processors. They have two distinct data streams:
    *   **Apical Dendrites (Top-Down):** Receive predictions from higher layers / context.
    *   **Basal Dendrites (Bottom-Up):** Receive sensory input / error signals.
*   **Function:** The neuron fires only when the *Prediction* does not match the *Input*. The spike represents **Surprise**.

### 2. PC-SNN (Lan et al., 2022)
**Source:** [arXiv:2211.15386](https://arxiv.org/abs/2211.15386)

*   **Insight:** We can perform Supervised Learning without Backpropagation.
*   **Mechanism:** Use **Local Hebbian Plasticity**.
    *   If Pre-Synaptic spike leads to Post-Synaptic spike -> Strengthen.
    *   The "Error" is encoded locally in the membrane potential voltage.

## Architecture Specification

### The "Soma" (Cell Body)
*   **Role:** Integration center.
*   **Equation:** `dV/dt = (V_rest - V) + I_basal + I_apical`

### The "Dendrite" (Branch)
*   **Role:** Non-linear computation of inputs.
*   **Property:** Active conductance (NMDA spikes). A dendrite can amplify a signal before it even hits the soma.

### The Network Topology
*   **Layer N:** Predicts state of Layer N-1.
*   **Layer N-1:** Sends error (Input - Prediction) to Layer N.
*   **Learning:** Minimize the firing rate of the Error Neurons. (If the brain is silent, it understands everything).
