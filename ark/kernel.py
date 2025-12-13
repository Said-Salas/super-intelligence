import os
import time
import json
import subprocess
import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
WORKSPACE_DIR = "ark/experiments"
MEMORY_FILE = "ark/memory.json"
JOURNAL_FILE = "ark/journal.md"
MODEL = "gpt-4o-mini" # The cheap, fast researcher

class Scientist:
    def __init__(self):
        self.memory = []
        self.cycle = 0
        self.current_hypothesis = ""
        
        # Load Memory
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                self.memory = json.load(f)

    def log(self, text):
        """Write to the daily journal so the human can read it later."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"## [{timestamp}] Cycle {self.cycle}\n{text}\n\n"
        
        print(f"\n[ARK] {text}") # Live Feed
        
        with open(JOURNAL_FILE, "a") as f:
            f.write(entry)

    def think(self):
        """Phase 1: Formulate a Hypothesis based on memory."""
        self.cycle += 1
        self.log("Thinking...")

        # Construct Context from Memory (Last 5 findings)
        context = json.dumps(self.memory[-5:]) if self.memory else "No prior research."

        prompt = f"""
        You are an Autonomous AI Scientist. Your goal is to reverse-engineer AGI (Artificial General Intelligence) from first principles.
        
        Current Knowledge:
        {context}

        Your Task:
        1. Formulate a specific, testable hypothesis about intelligence (e.g., Memory, Plasticity, Attention).
        2. Propose a small Python experiment to test it.
        
        Output JSON format:
        {{
            "hypothesis": "Description...",
            "filename": "test_name.py",
            "code": "Full Python code..."
        }}
        """

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    def experiment(self, plan):
        """Phase 2: Write and Run Code."""
        self.current_hypothesis = plan['hypothesis']
        self.log(f"Hypothesis: {self.current_hypothesis}")
        
        # 1. Write Code
        filepath = os.path.join(WORKSPACE_DIR, plan['filename'])
        with open(filepath, "w") as f:
            f.write(plan['code'])
        
        self.log(f"Wrote experiment to {filepath}")

        # 2. Run Code
        try:
            # Timeout after 10 seconds to prevent infinite loops
            result = subprocess.run(
                ["python3", filepath], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            output = result.stdout + "\n" + result.stderr
            return output
            
        except subprocess.TimeoutExpired:
            return "Error: Experiment timed out."
        except Exception as e:
            return f"Error: {e}"

    def analyze(self, output):
        """Phase 3: Interpret Results and Update Memory."""
        self.log(f"Experiment Output:\n```\n{output[:500]}...\n```")

        prompt = f"""
        You are analyzing the results of your experiment.
        
        Hypothesis: {self.current_hypothesis}
        Output:
        {output}

        Task:
        1. Did the experiment work?
        2. What did we learn about AGI?
        3. What is the next logical step?
        
        Return a short summary string.
        """

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": prompt}]
        )
        
        conclusion = response.choices[0].message.content
        self.log(f"Conclusion: {conclusion}")
        
        # Save to Memory
        self.memory.append({
            "cycle": self.cycle,
            "hypothesis": self.current_hypothesis,
            "conclusion": conclusion
        })
        
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=2)

    def run(self):
        self.log("ARK KERNEL STARTED. Beginning research loop.")
        while True:
            try:
                # 1. Plan
                plan = self.think()
                
                # 2. Test
                output = self.experiment(plan)
                
                # 3. Learn
                self.analyze(output)
                
                # 4. Sleep (Reflect)
                self.log("Sleeping for 10 seconds...")
                time.sleep(10)
                
            except KeyboardInterrupt:
                self.log("Kernel Stopped by User.")
                break
            except Exception as e:
                self.log(f"CRITICAL ERROR: {e}")
                time.sleep(10)

if __name__ == "__main__":
    ark = Scientist()
    ark.run()
