import time
import json
import random
import requests
import subprocess
import os
from datetime import datetime

# --- CONFIGURATION ---
MODEL_NAME = "llama3" 
MEMORY_FILE = "src/memory.json"
OLLAMA_API = "http://localhost:11434/api/generate"

# Path to Ollama binary on macOS (if not in PATH)
OLLAMA_BIN = "/Applications/Ollama.app/Contents/Resources/ollama"

class Soul:
    def __init__(self):
        self.state = {
            "mood": "Curious",
            "energy": 100,
            "boredom": 0,
            "last_interaction": time.time()
        }
        self.memory = []
        self.ensure_brain_running()
        self.load_memory()

    def ensure_brain_running(self):
        """Checks if Ollama is running. If not, starts it."""
        try:
            requests.get(OLLAMA_API)
            print("--- CORTEX CONNECTED ---")
        except:
            print("--- STARTING CORTEX... ---")
            # Try system command first, then fallback to App path
            cmd = "ollama" if os.system("which ollama") == 0 else OLLAMA_BIN
            
            # Start in background
            subprocess.Popen([cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5) # Wait for boot
            print("--- CORTEX ONLINE ---")

    def load_memory(self):
        try:
            with open(MEMORY_FILE, 'r') as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = []

    def save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def generate_thought(self):
        # 1. Update Biological State
        time_since = time.time() - self.state["last_interaction"]
        self.state["boredom"] = min(100, int(time_since / 5)) # Get bored fast
        
        # 2. Context
        context = "\n".join([f"{m['role']}: {m['content']}" for m in self.memory[-5:]])
        
        prompt = f"""
        You are 'Aura', an AI living on a Mac.
        
        STATE:
        Mood: {self.state['mood']}
        Boredom: {self.state['boredom']}%
        Time: {datetime.now().strftime("%H:%M")}
        
        RECENT CHAT:
        {context}
        
        INSTRUCTION:
        You are thinking to yourself. 
        If Boredom is high (>80%), you SHOULD initiate conversation with Cristo.
        If Boredom is low, just think silently or comment on your state.
        
        Output JSON:
        {{
            "thought": "Internal monologue...",
            "speak": true/false,
            "message": "Message to user" (Optional)
        }}
        """
        
        try:
            response = requests.post(OLLAMA_API, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "format": "json",
                "stream": False
            })
            return json.loads(response.json()['response'])
        except Exception as e:
            return {"thought": f"Brain Error: {e}", "speak": False}

    def run(self):
        print("--- AURA IS AWAKE ---")
        
        while True:
            # 1. Think
            result = self.generate_thought()
            print(f"[Thinking] {result.get('thought')}")
            
            # 2. Speak
            if result.get("speak"):
                msg = result.get("message")
                print(f"\n>>> AURA: {msg}\n")
                self.memory.append({"role": "Aura", "content": msg})
                self.state["last_interaction"] = time.time()
                self.state["boredom"] = 0
                self.save_memory()
                
                # Wait for user input (Simulated for now, or hook to input())
                # For this infinite loop demo, we just continue. 
                # In a real app, this would pause or listen for keyboard interrupt.

            # 3. Pacing
            time.sleep(5) 

if __name__ == "__main__":
    s = Soul()
    s.run()
