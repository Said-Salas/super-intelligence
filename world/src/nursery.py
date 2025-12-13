import pygame
import pymunk
import pymunk.pygame_util
import math
import random

# --- CONFIGURATION ---
WIDTH, HEIGHT = 1000, 600
FPS = 60

# --- NEUROSCIENCE LAYER (The Spiking Brain) ---
class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.potential = 0.0
        self.threshold = 1.0
        self.decay = 0.90
        self.synapses = [] 
        self.refractory = 0
        self.last_fired_time = -1

    def connect(self, target, weight):
        self.synapses.append({"target": target, "weight": weight})

    def receive_spike(self, current):
        if self.refractory <= 0:
            self.potential += current

    def tick(self, current_time):
        self.potential *= self.decay
        if self.refractory > 0:
            self.refractory -= 1
            return 0
        if self.potential >= self.threshold:
            self.fire()
            self.potential = 0.0
            self.refractory = 5
            self.last_fired_time = current_time
            return 1
        return 0

    def fire(self):
        for s in self.synapses:
            s["target"].receive_spike(s["weight"])

class SpikingBrain:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Layers
        self.sensory = [Neuron(i) for i in range(input_size)]
        self.inter = [Neuron(i+input_size) for i in range(30)]
        self.motor = [Neuron(i+input_size+30) for i in range(output_size)]
        
        self.all_neurons = self.sensory + self.inter + self.motor
        self.time = 0

        # Wiring
        for n in self.all_neurons:
            targets = random.sample(self.all_neurons, 4)
            for t in targets:
                if t != n:
                    # Initial weights: mix of excitatory/inhibitory
                    n.connect(t, random.uniform(-0.5, 1.2))

    def process(self, inputs):
        self.time += 1
        
        # 1. Noise (Spontaneous Activity)
        for n in self.all_neurons:
            if random.random() < 0.2: # 20% chance
                n.receive_spike(1.5) # Strong jolt (Guaranteed spike)

        # 2. Sensory Input
        for i, val in enumerate(inputs):
            # val is 0.0 to 1.0. Convert to current.
            self.sensory[i].receive_spike(val * 2.0)

        # 3. Processing Ticks
        for _ in range(4):
            for n in self.all_neurons:
                n.tick(self.time)

        # 4. Motor Output (Accumulate spikes)
        commands = []
        for n in self.motor_neurons_list():
            # If refractory is high, it just fired
            if n.refractory > 3: 
                commands.append(1.0)
            else:
                commands.append(0.0)
        return commands

    def motor_neurons_list(self):
        return self.motor

    def train(self, reward):
        print(f"Dopamine: {reward}")
        for n in self.all_neurons:
            if self.time - n.last_fired_time < 30:
                for s in n.synapses:
                    s["weight"] += reward * 0.1
                    s["weight"] = max(-2.0, min(3.0, s["weight"]))

# --- PHYSICS LAYER (The Ragdoll) ---
class Walker:
    def __init__(self, space, x, y):
        self.space = space
        self.bodies = []
        
        # 1. Torso
        self.torso_body = pymunk.Body(10, float("inf")) # Mass 10
        self.torso_body.position = (x, y)
        self.torso_shape = pymunk.Poly.create_box(self.torso_body, (30, 60))
        self.torso_shape.color = (0, 0, 255, 255)
        self.torso_shape.friction = 0.5
        space.add(self.torso_body, self.torso_shape)
        self.bodies.append(self.torso_body)

        # 2. Left Leg (Thigh)
        self.l_thigh, self.l_hip_motor = self.create_limb(x, y+30, 10, 40, self.torso_body, (0, 20), -1)
        # 3. Right Leg (Thigh)
        self.r_thigh, self.r_hip_motor = self.create_limb(x, y+30, 10, 40, self.torso_body, (0, 20), 1)

    def create_limb(self, x, y, w, h, parent_body, anchor, side_offset):
        body = pymunk.Body(2, float("inf"))
        body.position = (x + (side_offset * 10), y)
        shape = pymunk.Poly.create_box(body, (w, h))
        shape.friction = 0.8
        shape.color = (0, 255, 0, 255)
        self.space.add(body, shape)
        
        # Joint (Pivot)
        joint = pymunk.PivotJoint(parent_body, body, parent_body.position + pymunk.Vec2d(side_offset * 5, 30))
        joint.collide_bodies = False
        
        # Motor (Muscle)
        motor = pymunk.SimpleMotor(parent_body, body, 0)
        motor.max_force = 100000 # Strength
        
        self.space.add(joint, motor)
        return body, motor

    def get_state(self):
        # Proprioception: What does the body feel?
        # 1. Torso Angle (Normalized -1 to 1)
        angle = self.torso_body.angle
        # 2. Angular Velocity
        ang_vel = self.torso_body.angular_velocity
        # 3. Height from ground
        height = 600 - self.torso_body.position.y
        
        return [
            math.tanh(angle), 
            math.tanh(ang_vel), 
            min(1.0, height / 200.0)
        ]

    def apply_force(self, motor_commands):
        # Motor commands: [Left Hip Forward, Left Hip Back, Right Hip Forward, Right Hip Back]
        # We map 4 outputs to 2 motors
        
        # Speed Multiplier: 20 (Strong kick)
        SPEED = 20.0
        
        # Left Hip
        l_power = (motor_commands[0] - motor_commands[1]) * SPEED
        self.l_hip_motor.rate = l_power
        
        # Right Hip
        r_power = (motor_commands[2] - motor_commands[3]) * SPEED
        self.r_hip_motor.rate = r_power

# --- MAIN SIMULATION ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepMind Nursery: Learning to Walk")
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Physics Space
    space = pymunk.Space()
    space.gravity = (0.0, 900.0)

    # Floor
    floor = pymunk.Body(body_type=pymunk.Body.STATIC)
    floor_shape = pymunk.Segment(floor, (0, HEIGHT-20), (WIDTH, HEIGHT-20), 5)
    floor_shape.friction = 1.0
    space.add(floor, floor_shape)

    # Agent
    walker = Walker(space, 300, 300)
    
    # Brain (Inputs: 3 Body + 1 Vision, Outputs: 4 Muscles)
    brain = SpikingBrain(input_size=4, output_size=4)

    # Target (Toy)
    toy_x = 800

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    brain.train(1.0)
                elif event.key == pygame.K_DOWN:
                    brain.train(-1.0)
                elif event.key == pygame.K_SPACE:
                    # Reset
                    walker.torso_body.position = (300, 300)
                    walker.torso_body.angle = 0
                    walker.torso_body.velocity = (0,0)

        # 1. Sense
        body_state = walker.get_state()
        # Vision: Distance to toy
        dist_to_toy = (toy_x - walker.torso_body.position.x) / 500.0
        inputs = body_state + [dist_to_toy]

        # 2. Think
        motor_outputs = brain.process(inputs)

        # 3. Act
        walker.apply_force(motor_outputs)

        # Physics Step
        space.step(1/FPS)

        # Draw
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        
        # Draw Toy
        pygame.draw.circle(screen, (255, 0, 0), (toy_x, HEIGHT-40), 20)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
