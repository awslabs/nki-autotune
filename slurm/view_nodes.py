import os
import subprocess


def get_slurm_nodes():
    if "SLURM_JOB_NODELIST" in os.environ:
        cmd = f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        nodes = result.stdout.strip().split("\n")
    else:
        nodes = ["localhost"]
    return nodes


nodes = get_slurm_nodes()
print(nodes)
