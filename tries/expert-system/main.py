import time
import os
import sys
#Not AGI.
class SuperIntelligence:
    def __init__(self):
        self.is_running = True
        self.cycle_count = 0
        self.memory = []

    def perceive(self):
        timestamp = time.time()
        files = os.listdir('.')
        return {
            "timestamp": timestamp,
            "cycle": self.cycle_count,
            "files": files
        }

    def update_state(self, sensory_input):
        self.memory.append(sensory_input)
        if len(self.memory) > 10:
            self.memory.pop(0)

    def think(self, internal_state):
        current_files = set(internal_state["files"])

        if len(self.memory) > 1:
            previous_files = set(self.memory[-2]["files"])

            new_files = current_files - previous_files

            if new_files:
                filename = list(new_files)[0]

                try:
                    with open(filename, 'r') as f: 
                        content = f.read()

                    reply = f"I processed your file: {filename}. You said: {content}"

                    return {
                        "action": "create_file", 
                        "filename": f"reply_to_{filename}",
                        "content": reply
                    }   

                except Exception as e:
                    return {"action": "log_status", "data": f"Failed to read {filename}: {e}"}    
            
            elif len(current_files) < len(previous_files):
                return {"action": "log_status", "data": "File deleted!"}

        return {"action": "wait", "reason": "no_change"}

    def act(self, plan):
        if plan["action"] == "create_file":
            with open(plan["filename"], 'w') as f:
                f.write(plan["content"])
            print(f"Cycle {self.cycle_count}: WROTE RESPONSE to {plan['filename']}")

        elif plan["action"] == "wait":
            print(f"Cycle {self.cycle_count}: Waiting...")

    def run(self):
        while self.is_running:
            sensory_input = self.perceive()
            self.update_state(sensory_input)
            plan = self.think(sensory_input)
            self.act(plan)

            self.cycle_count += 1
            time.sleep(8)

if __name__ == "__main__":
    ai = SuperIntelligence()
    ai.run()
