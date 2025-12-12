import time
import os
import sys

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
        current_files = internal_state["files"]

        if len(self.memory) > 1:
            previous_files = self.memory[-2]["files"]

            if len(current_files) > len(previous_files):
                return {"action": "log_status", "data": "New file detected"}
            
            elif len(current_files) < len(previous_files):
                return {"action": "log_status", "data": "File deleted!"}

        return {"action": "wait", "reason": "no_change"}

    def act(self, plan):
        if plan["action"] == "wait":
            print(f"Cycle {self.cycle_count}: Waiting...")
        elif plan["action"] == "log_status":
            print(f"Cycle {self.cycle_count}: ALERT - {plan['data']}")

    def run(self):
        while self.is_running:
            sensory_input = self.perceive()
            self.update_state(sensory_input)
            plan = self.think(sensory_input)
            self.act(plan)

            self.cycle_count += 1
            time.sleep(4)

if __name__ == "__main__":
    ai = SuperIntelligence()
    ai.run()
