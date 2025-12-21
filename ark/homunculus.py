import os
import time
import json
import base64
import platform
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# OS-Specific Libraries (Install these on the VM)
try:
    import pyautogui
    import mss
except ImportError:
    print("Missing libraries. Run: pip install pyautogui mss")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIGURATION
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
HISTORY_FILE = "ark/life_log.md"

class DigitalHomunculus:
    def __init__(self):
        self.goal = "Research the nature of forgetting in neural networks by browsing ArXiv.org and taking notes."
        self.memory = []
        self.screenshot_path = "ark/vision.png"
        
        # Safety: Fail-safe (Move mouse to corner to kill)
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 1.0 # Slow down actions to be human-like

    def see(self):
        """Capture the screen."""
        with mss.mss() as sct:
            # Capture full screen
            monitor = sct.monitors[1] 
            sct.shot(mon=1, output=self.screenshot_path)
            
        with open(self.screenshot_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def think(self, b64_image):
        """Decide the next mouse/keyboard action."""
        
        prompt = f"""
        You are a Digital Entity living inside a computer.
        Your current High-Level Goal is: "{self.goal}"
        
        You have full control of the mouse and keyboard.
        The screen resolution is {SCREEN_WIDTH}x{SCREEN_HEIGHT}.
        
        Analyze the screenshot. What is the IMMEDIATE next step?
        
        Output JSON ONLY:
        {{
            "thought": "I need to open the browser...",
            "action": "click" | "double_click" | "type" | "press" | "wait",
            "x": 0, (only for clicks)
            "y": 0, (only for clicks)
            "text": "arxiv.org", (only for type)
            "key": "enter" (only for press)
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a computer operator. Output valid JSON only."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                    ]
                }
            ],
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    def act(self, plan):
        """Execute the physical action."""
        action = plan.get('action')
        thought = plan.get('thought')
        
        self.log(f"Thinking: {thought}")
        self.log(f"Action: {action}")

        try:
            if action == "click":
                x, y = plan['x'], plan['y']
                pyautogui.moveTo(x, y, duration=0.5)
                pyautogui.click()
                
            elif action == "double_click":
                x, y = plan['x'], plan['y']
                pyautogui.moveTo(x, y, duration=0.5)
                pyautogui.doubleClick()
                
            elif action == "type":
                text = plan['text']
                pyautogui.write(text, interval=0.1) # Type like a human
                
            elif action == "press":
                key = plan['key']
                pyautogui.press(key)
                
            elif action == "wait":
                time.sleep(2)
                
        except Exception as e:
            self.log(f"Motor Control Error: {e}")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        with open(HISTORY_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def run(self):
        self.log("--- HOMUNCULUS AWAKENED ---")
        self.log(f"Goal: {self.goal}")
        
        # Give user time to switch windows or cancel
        print("Starting in 5 seconds... Press Ctrl+C to Abort.")
        time.sleep(5)
        
        while True:
            try:
                # 1. Perception
                vision = self.see()
                
                # 2. Cognition
                plan = self.think(vision)
                
                # 3. Action
                self.act(plan)
                
                # 4. Reflex Loop Speed
                time.sleep(1) # Wait a bit between actions to verify result
                
            except KeyboardInterrupt:
                self.log("Shutting down.")
                break
            except Exception as e:
                self.log(f"CRITICAL ERROR: {e}")
                time.sleep(5)

if __name__ == "__main__":
    entity = DigitalHomunculus()
    entity.run()
