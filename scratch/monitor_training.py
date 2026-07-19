import subprocess
import time
import os
import sys

log_path = "/Users/govind/.gemini/antigravity/brain/73ebeb87-8153-485a-aa31-63bf6bcc7b79/.system_generated/tasks/task-1133.log"

print("Starting training monitor daemon for 5 epochs limit...")
while True:
    try:
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                content = f.read()
                # If Epoch 5 starts, it means Epoch 0, 1, 2, 3, 4 (5 epochs) have finished and saved!
                if "Epoch: 5" in content or "Epoch: 5 | Batch: 0/1003" in content:
                    print("Epoch 5 initialization detected (5 completed epochs). Terminating training process...")
                    # Find and kill the train_htr.py process
                    ps_output = subprocess.check_output(["ps", "aux"]).decode("utf-8")
                    killed = False
                    for line in ps_output.split("\n"):
                        if "train_htr.py" in line and "python" in line and "grep" not in line and "monitor" not in line:
                            parts = line.split()
                            if len(parts) > 1:
                                pid = parts[1]
                                print(f"Killing training process PID {pid}...")
                                subprocess.call(["kill", pid])
                                killed = True
                    if killed:
                        print("Training process terminated successfully. Exiting monitor.")
                        sys.exit(0)
    except Exception as e:
        print(f"Error in monitor: {e}")
    time.sleep(15)
