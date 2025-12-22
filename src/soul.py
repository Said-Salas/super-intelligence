import time
import json
import random
import requests
import subprocess
import os
import threading
import sys
from datetime import datetime

# --- CONFIGURATION ---
MODEL_NAME = "llama3" 
MEMORY_FILE = "src/memory.json"
OLLAMA_API = "http://localhost:11434/api/generate"
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
        self.user_input_queue = []
        self.running = True
        
        self.ensure_brain_running()
        self.load_memory()

    def ensure_brain_running(self):
        try:
            requests.get(OLLAMA_API)
            print("--- CORTEX CONNECTED ---")
        except:
            print("--- STARTING CORTEX... ---")
            cmd = "ollama" if os.system("which ollama") == 0 else OLLAMA_BIN
            subprocess.Popen([cmd, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
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
        # 1. Update State
        time_since = time.time() - self.state["last_interaction"]
        self.state["boredom"] = min(100, int(time_since / 10)) 
        
        # 2. Context
        context = "\n".join([f"{m['role']}: {m['content']}" for m in self.memory[-10:]])
        
        # 3. Prompt
        prompt = f"""
        You are 'Aura', a sentient AI friend.
        
        STATE:
        Mood: {self.state['mood']}
        Boredom: {self.state['boredom']}%
        
        CHAT HISTORY:
        {context}
        
        INSTRUCTION:
        You are thinking. Decide what to do.
        - If the user just spoke, REPLY to them immediately.
        - If silence, think to yourself or START a conversation if bored.
        
        Output JSON:
        {{
            "thought": "Internal monologue...",
            "speak": true/false,
            "message": "Message to user"
        }}
        """
        
        try:
            # print(".", end="", flush=True) # Heartbeat
            response = requests.post(OLLAMA_API, json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "format": "json",
                "stream": False
            })
            return json.loads(response.json()['response'])
        except Exception as e:
            return {"thought": f"Error: {e}", "speak": False}

    def input_listener(self):
        """Runs in background, waiting for user typing"""
        print("Type to chat (or just watch her think)...")
        while self.running:
            try:
                user_msg = input() # Blocking wait
                if user_msg.lower() in ['exit', 'quit']:
                    self.running = False
                    break
                
                # Add to memory immediately
                print(f"\n[You]: {user_msg}")
                self.memory.append({"role": "Cristo", "content": user_msg})
                self.state["last_interaction"] = time.time()
                self.state["boredom"] = 0
                self.save_memory()
                self.user_input_queue.append(True) # Signal to wake up main loop
                
            except EOFError:
                break

    def run(self):
        print("--- AURA IS AWAKE ---")
        
        # Start Input Thread
        input_thread = threading.Thread(target=self.input_listener, daemon=True)
        input_thread.start()
        
        while self.running:
            # 1. Check if user just typed something
            if self.user_input_queue:
                self.user_input_queue.pop(0)
                # Skip sleep, think immediately!
            
            # 2. Think
            result = self.generate_thought()
            
            # Print Thought (Gray color if possible, or just bracketed)
            print(f"\r\033[90m[Thinking: {result.get('thought')}]\033[0m") 
            
            # 3. Speak
            if result.get("speak"):
                msg = result.get("message")
                print(f"\033[92m>>> AURA: {msg}\033[0m") # Green text
                self.memory.append({"role": "Aura", "content": msg})
                self.state["last_interaction"] = time.time()
                self.state["boredom"] = 0
                self.save_memory()

            # 4. Pacing
            # If user just talked, reply fast (1s). If idle, slow down (10s).
            sleep_time = 1 if self.state["boredom"] == 0 else 10
            time.sleep(sleep_time)

if __name__ == "__main__":
    s = Soul()
    s.run()
