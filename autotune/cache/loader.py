import json
import shutil
import subprocess
from typing import Dict

from autotune.cache.directories import split_file_info


def get_best_result(json_file: str) -> Dict:
    with open(json_file, "r") as f:
        data = json.load(f)
    best_result = data["results"][0]
    return best_result


def profile_upload(neff: str):
    directory, neff_name, file_type = split_file_info(neff)
    num_iters = 10
    assert file_type == "neff", f"{neff} is not a .neff file."
    ntff_file = f"{directory}/{neff_name}.ntff"
    trace_cmd = f"neuron-profile capture -n {neff} --profile-nth-exec={num_iters}"
    subprocess.run(trace_cmd, shell=True)
    shutil.move(f"profile_exec_{num_iters}.ntff", ntff_file)
    upload_command = f'profile-upload -F "neff=@{neff_name}.neff" -F "ntff=@{neff_name}.ntff" -F name={neff_name}'
    print(upload_command)
