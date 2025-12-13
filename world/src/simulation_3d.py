from ursina import *
import threading
import time
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- CLOUD VISION LAYER ---
class CloudEye:
    def __init__(self):
        self.is_processing = False
        self.last_thought = "Waiting for visual input..."
        self.last_seen_time = 0
        self.client = None
        
        # Initialize Client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            print("--- CORTEX ONLINE: Connected to OpenAI ---")
        else:
            print("--- CORTEX OFFLINE: No API Key found in .env ---")

    def capture_and_think(self):
        """
        Takes a screenshot and sends it to the Cloud Brain.
        Runs in a background thread to avoid freezing the game.
        """
        if self.is_processing:
            return
        
        self.is_processing = True
        
        # 1. Capture the Screen (Agent's POV)
        # We save the current frame to a file
        filename = "agent_view.png"
        application.base.win.saveScreenshot(filename)
        
        # Start the heavy lifting in a thread
        threading.Thread(target=self._send_to_api, args=(filename,), daemon=True).start()

    def _send_to_api(self, filename):
        try:
            # Wait a moment for file to be written
            time.sleep(0.5)
            
            if not self.client:
                # Mock Mode if no API key
                time.sleep(1)
                thought = "I see a simulation. I need an API Key to wake up."
            else:
                # 2. Encode Image
                with open(filename, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                print("--- UPLOADING TO CLOUD ---")
                
                # 3. The API Call
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an autonomous AI agent in a 3D simulation. Briefly describe what you see and decide your next action. Format: 'I see [objects]. I will [action]'."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is your next move?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )
                thought = response.choices[0].message.content
            
            self.last_thought = thought
            self.last_seen_time = time.time()
            print(f"CLOUD BRAIN SAYS: {thought}")
            
        except Exception as e:
            print(f"API ERROR: {e}")
            self.last_thought = f"Error: {e}"
        
        finally:
            self.is_processing = False

# --- THE 3D WORLD ---
app = Ursina()

# 1. The Environment
ground = Entity(model='plane', scale=(50, 1, 50), color=color.gray.tint(-.2), texture='white_cube', texture_scale=(50,50), collider='box')
walls = [
    Entity(model='cube', position=(0,2.5,25), scale=(50,5,1), color=color.gray, collider='box'),
    Entity(model='cube', position=(0,2.5,-25), scale=(50,5,1), color=color.gray, collider='box'),
    Entity(model='cube', position=(25,2.5,0), scale=(1,5,50), color=color.gray, collider='box'),
    Entity(model='cube', position=(-25,2.5,0), scale=(1,5,50), color=color.gray, collider='box')
]

# 2. Objects
ball = Entity(model='sphere', color=color.red, scale=(2,2,2), position=(5, 1, 10), collider='sphere')
cube = Entity(model='cube', color=color.green, scale=(2,2,2), position=(-5, 1, 10), collider='box')

# 3. The Agent
agent = Entity(model='capsule', color=color.white, scale=(1,2,1), position=(0, 1, 0), collider='capsule')
head = Entity(parent=agent, position=(0, 0.8, 0))

# 4. The Cloud Brain connection
eye = CloudEye()

# GUI
thought_bubble = Text(text="Booting...", position=(-0.85, 0.45), scale=1.5, color=color.yellow, background=True)
status_light = Entity(parent=camera.ui, model='quad', scale=(0.05, 0.05), position=(0.85, 0.45), color=color.green)

def update():
    # --- 1. VISION LOOP ---
    # Trigger a new thought every 8 seconds (Automatic)
    if time.time() - eye.last_seen_time > 8.0:
        eye.capture_and_think()

    # Update GUI
    thought_bubble.text = eye.last_thought
    status_light.color = color.red if eye.is_processing else color.green

    # --- 2. MOTOR LOOP (Reaction) ---
    # Simple semantic parsing of the thought
    speed = 6 * time.dt
    rot_speed = 100 * time.dt
    
    thought = eye.last_thought.lower()
    
    if "red" in thought or "ball" in thought or "sphere" in thought:
        agent.look_at_2d(ball.position, 'y')
        agent.position += agent.forward * speed
    
    elif "green" in thought or "cube" in thought:
        agent.look_at_2d(cube.position, 'y')
        agent.position += agent.forward * speed
        
    elif "wall" in thought or "turn" in thought:
        agent.rotation_y += rot_speed
    
    else:
        # Wander
        agent.rotation_y += rot_speed * 0.1

    # Physics
    agent.x = clamp(agent.x, -24, 24)
    agent.z = clamp(agent.z, -24, 24)

# Camera Setup
EditorCamera()
camera.position = (0, 10, -20)
camera.look_at(agent)

app.run()
