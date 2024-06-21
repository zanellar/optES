import subprocess
import time
import wandb
import os

num_processes = 14
conda_env = 'nes_pendulum'

###########################################################################################

# Keep track of the Popen objects
processes = []

# Define the command to open a new terminal, activate conda environment and run a script
command = f'gnome-terminal -- bash -c  "source activate {conda_env}; python run/pendulum/sweep_pendulum_naespbc.py"' 

# Run the command
p0 = subprocess.Popen(command, shell=True)
processes.append(p0)

###########################################################################################

sweep_id = str(input("Enter sweep id: "))

# Open python file and modifiy sweep_id
with open('run/pendulum/sweep_pendulum_naespbc.py', 'r') as file:
    filedata = file.read()

# Replace the sweep_id  
filedata = filedata.replace("sweep_id = wandb.sweep(sweep = config, project = project)", f"sweep_id = str('{sweep_id}')")

# Remove baselineclass 
filedata = filedata.replace("baselineclass = PendulumASEPBC,", "")


filedata = filedata.replace("wandb.finish()", "")

# Write in a new file
with open('run/pendulum/_tmp.py', 'w') as file:
    file.write(filedata)

for i in range(num_processes):
        
    # Define the command to open a new terminal, activate conda environment and run a script
    command =  f'gnome-terminal -- bash -c  "source activate {conda_env}; python run/pendulum/_tmp.py"' 

    # Run the command
    pi = subprocess.Popen(command, shell=True)
    processes.append(pi)

    # Wait 5 seconds
    time.sleep(10)
 
# delete the output file
input("Press Enter to kill...")
# Kill the processes and close the terminals
for p in processes:
    p.terminate()
os.remove('run/pendulum/_tmp.py') 