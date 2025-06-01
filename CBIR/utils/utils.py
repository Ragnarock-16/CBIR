import os

def get_new_run_folder(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    run_ids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_id = max(run_ids, default=0) + 1
    run_folder = os.path.join(base_dir, f"run_{next_id}")
    os.makedirs(run_folder)
    return run_folder