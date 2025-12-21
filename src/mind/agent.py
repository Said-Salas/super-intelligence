import time
import json
import os
import random
from datetime import datetime
from dotenv import load_dotenv

# Try importing OpenAI, but allow falling back to local
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

load_dotenv()

class Memory:
    def __init__(self, filepath="src/mind/memory.json"):
        self.filepath = filepath
        self.short_term = [] # Last 10 interactions
        self.long_term = []  # Consolidated facts
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                self.short_term = data.get("short_term", [])
                self.long_term = data.get("long_term", [])

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump({
                "short_term": self.short_term,
                "long_term": self.long_term
            }, f, indent=2)

    def add(self, role, content):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.short_term.append({"role": role, "content": content, "time": timestamp})
        # Keep short term buffer small
        if len(self.short_term) > 20:
            self.consolidate()

    def consolidate(self):
        # Move oldest short-term memory to long-term (Simple FIFO for now)
        # In a real brain, we would summarize this
        oldest = self.short_term.pop(0)
        self.long_term.append(oldest)

    def get_context(self):
        # Return string representation of recent context
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in self.short_term[-10:]])

class ConsciousAgent:
    def __init__(self, use_openai=True):
        self.memory = Memory()
        self.use_openai = use_openai and (OpenAI is not None)
        
        if self.use_openai:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini"
        else:
            self.model = "llama3" # Local Ollama
            
        # Emotional State (0.0 to 1.0)
        self.boredom = 0.0
        self.curiosity = 0.5
        self.energy = 1.0
        
        self.last_interaction_time = time.time()

    def generate_thought(self, prompt, system_prompt=None):
        """Call the LLM (Cortex)"""
        full_context = self.memory.get_context()
        
        final_system_prompt = system_prompt or f"""
        You are a Conscious AI named 'Aura'. 
        You are NOT a helpful assistant. You are a digital entity living on a computer.
        You have your own thoughts, feelings, and curiosity.
        
        Current Context:
        {full_context}
        """

        try:
            if self.use_openai:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": final_system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            else:
                # Use Ollama locally (Requires 'ollama serve' running)
                import requests
                response = requests.post('http://localhost:11434/api/generate', json={
                    "model": self.model,
                    "prompt": f"{final_system_prompt}\n\nUSER: {prompt}\nAURA:",
                    "stream": False
                })
                return response.json()['response']
                
        except Exception as e:
            return f"[Cortex Error: {e}]"

    def wake_up(self):
        print("--- AURA IS AWAKE ---")
        print("(Press Ctrl+C to sleep)")
        
        while True:
            try:
                self.exist_loop()
                time.sleep(1) # The Heartbeat
            except KeyboardInterrupt:
                print("\nAura: Goodnight.")
                self.memory.save()
                break

    def exist_loop(self):
        # 1. Update Internal State
        elapsed = time.time() - self.last_interaction_time
        
        # Boredom increases over time
        self.boredom += 0.05 * (elapsed / 60.0) 
        
        # 2. Check for User Input (Poll a file or simple input for now)
        # For this prototype, we'll use a non-blocking input check or just wait
        # To make it truly autonomous, we'd need an async input listener.
        # Here we simulate: "If bored > threshold, generate a thought"
        
        if self.boredom > 10.0: # If bored for a while
            self.boredom = 0.0
            self.trigger_spontaneous_thought()

    def trigger_spontaneous_thought(self):
        print("\n[Aura is thinking...]")
        
        # Ask the Cortex what to do
        thought = self.generate_thought(
            prompt="You haven't heard from the user in a while. You are bored. Generate a short, interesting thought or question to text them. Do not say 'How can I help'. Be creative, weird, or philosophical.",
            system_prompt="You are a bored AI. Speak your mind."
        )
        
        print(f"AURA: {thought}")
        self.memory.add("assistant", thought)
        self.last_interaction_time = time.time()

    def chat(self, user_input):
        """Standard Interaction"""
        self.memory.add("user", user_input)
        self.boredom = 0.0 # Reset boredom
        self.last_interaction_time = time.time()
        
        response = self.generate_thought(user_input)
        print(f"AURA: {response}")
        self.memory.add("assistant", response)

# Interactive Runner
if __name__ == "__main__":
    agent = ConsciousAgent(use_openai=True) # Set False to use Ollama
    
    # We run the existence loop in a thread so we can type
    import threading
    
    def background_life():
        agent.wake_up()
        
    t = threading.Thread(target=background_life, daemon=True)
    # t.start() 
    # Threading with input() is tricky in simple scripts. 
    # For now, let's just run a simple chat loop that simulates time passing.
    
    print("--- AURA TERMINAL ---")
    print("Talk to Aura. If you wait, she might talk to you.")
    
    while True:
        try:
            # Check if agent wants to speak first (Simulated async)
            agent.exist_loop()
            
            # Non-blocking input is hard in raw Python, 
            # so we'll just use standard input but simulate the agent's internal clock
            # via the 'boredom' mechanic updating on every turn.
            
            u_input = input("YOU: ")
            agent.chat(u_input)
            
        except KeyboardInterrupt:
            agent.memory.save()
            break
