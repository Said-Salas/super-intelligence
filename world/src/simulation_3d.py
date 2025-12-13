from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
import random
import math

# --- SPIKING NEURAL NETWORK (The Brain) ---
class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.potential = 0.0
        self.threshold = 1.0
        self.decay = 0.95
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
            self.refractory = 10
            self.last_fired_time = current_time
            return 1
        return 0

    def fire(self):
        for s in self.synapses:
            s["target"].receive_spike(s["weight"])

class SpikingBrain:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.sensory = [Neuron(i) for i in range(input_size)]
        self.inter = [Neuron(i+input_size) for i in range(20)]
        self.motor = [Neuron(i+input_size+20) for i in range(output_size)]
        self.all_neurons = self.sensory + self.inter + self.motor
        self.time = 0

        # Wiring
        for n in self.all_neurons:
            targets = random.sample(self.all_neurons, 3)
            for t in targets:
                if t != n:
                    n.connect(t, random.uniform(-0.5, 1.5))

    def process(self, inputs):
        self.time += 1
        
        # 1. Noise (Spontaneous Activity - Confusion/Exploration)
        for n in self.all_neurons:
            if random.random() < 0.2: # 20% chance of random thought
                n.receive_spike(1.2)

        # 2. Sensory Input
        for i, val in enumerate(inputs):
            self.sensory[i].receive_spike(val * 2.0)

        # 3. Think
        for _ in range(3):
            for n in self.all_neurons:
                n.tick(self.time)

        # 4. Motor Output
        return [1 if n.refractory > 5 else 0 for n in self.motor]

    def train(self, reward):
        print(f"DOPAMINE: {reward}")
        for n in self.all_neurons:
            if self.time - n.last_fired_time < 30:
                for s in n.synapses:
                    s["weight"] += reward * 0.1
                    s["weight"] = max(-1.0, min(3.0, s["weight"]))

# --- THE 3D WORLD ---
app = Ursina()

# 1. The Room (Physics Enabled)
ground = Entity(model='plane', scale=(50, 1, 50), color=color.gray.tint(-.2), texture='white_cube', texture_scale=(50,50), collider='box')
wall_n = Entity(model='cube', position=(0,2.5,25), scale=(50,5,1), color=color.gray, collider='box')
wall_s = Entity(model='cube', position=(0,2.5,-25), scale=(50,5,1), color=color.gray, collider='box')
wall_e = Entity(model='cube', position=(25,2.5,0), scale=(1,5,50), color=color.gray, collider='box')
wall_w = Entity(model='cube', position=(-25,2.5,0), scale=(1,5,50), color=color.gray, collider='box')

# 2. The Ball (Dynamic Physics Object)
ball = Entity(model='sphere', color=color.red, scale=(2,2,2), position=(5, 1, 5), collider='sphere')
ball_velocity = Vec3(0,0,0)

# 3. The Agent (Blue Sphere)
agent = Entity(model='sphere', color=color.blue, scale=(1.5,1.5,1.5), position=(0, 0.75, 0), collider='sphere')
# Eyes (Rays)
eyes = []
for i in range(5):
    eyes.append(Entity(model='cube', scale=(0.05, 0.05, 10), color=color.cyan, parent=agent, position=(0,0,0), origin_z=-0.5))
    eyes[i].visible = True # Visualizing the "Sight"

# 4. The Brain
brain = SpikingBrain(input_size=5, output_size=2)

def update():
    global ball_velocity
    
    # --- 1. SENSE (Lidar) ---
    input_vector = []
    
    # Cast 5 rays
    angles = [-30, -15, 0, 15, 30]
    for i, angle in enumerate(angles):
        # Ray direction in World Space
        # Calculate local direction then rotate by agent's Y rotation
        rad = math.radians(agent.rotation_y + angle)
        ray_dir = Vec3(math.sin(rad), 0, math.cos(rad))
        
        hit_info = raycast(agent.position, ray_dir, distance=20, ignore=(agent, eyes[i]))
        
        # Visualize Ray
        eyes[i].rotation_y = angle # Relative rotation to parent
        
        if hit_info.hit:
            dist = hit_info.distance
            eyes[i].scale_z = dist
            eyes[i].color = color.red # I see something!
            input_vector.append(1.0 - (dist / 20.0))
        else:
            eyes[i].scale_z = 10
            eyes[i].color = color.cyan # Clear
            input_vector.append(0.0)

    # --- 2. THINK ---
    motor_outputs = brain.process(input_vector)
    
    # --- 3. ACT ---
    # Tank Controls
    move_speed = 8 * time.dt
    turn_speed = 100 * time.dt
    
    left_motor = motor_outputs[0]
    right_motor = motor_outputs[1]
    
    if left_motor and right_motor:
        # Move Forward (Push)
        agent.position += agent.forward * move_speed
    elif left_motor:
        # Turn Right
        agent.rotation_y += turn_speed
    elif right_motor:
        # Turn Left
        agent.rotation_y -= turn_speed
        
    # --- 4. PHYSICS (First Principles) ---
    
    # Agent Collision with Walls
    # (Simple clamping for now, but feels physical)
    agent.x = clamp(agent.x, -24, 24)
    agent.z = clamp(agent.z, -24, 24)
    
    # Agent Pushing Ball (Elastic Collision)
    dist = distance(agent, ball)
    if dist < (agent.scale_x/2 + ball.scale_x/2):
        # Collision vector
        direction = (ball.position - agent.position).normalized()
        # Transfer momentum
        ball_velocity += direction * move_speed * 50 # Kick it
        # Push agent back slightly (Newton's 3rd Law)
        agent.position -= direction * move_speed 
        
        # Learning Event: Touching the ball is interesting!
        brain.train(0.5) 

    # Ball Physics (Friction & Movement)
    ball.position += ball_velocity * time.dt
    ball_velocity *= 0.98 # Friction
    
    # Ball Wall Bounce
    if ball.x > 24 or ball.x < -24: ball_velocity.x *= -1
    if ball.z > 24 or ball.z < -24: ball_velocity.z *= -1
    
    # Check "Goal" (Pushing ball to a specific area?) 
    # For now, we reward just moving it fast (Playfulness)
    if ball_velocity.length() > 2:
        brain.train(0.1)

def input(key):
    if key == 'up arrow':
        brain.train(1.0) # Good boy
        print("TEACHER: Good!")
    if key == 'down arrow':
        brain.train(-1.0) # Bad boy
        print("TEACHER: Bad!")
    if key == 'escape':
        application.quit()

# Camera (The Angel)
EditorCamera() 
camera.position = (0, 30, -30) # Bird's eye view start
camera.look_at(agent)

app.run()
