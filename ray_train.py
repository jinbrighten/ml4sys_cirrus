import json
import os
import subprocess

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch

from inference import inference_directory

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

RAW_F1 = 0.8
F1_THRESHOLD = 0.1

WIDTH_COUNT = 6
WIDTH_MIN = 10
WIDTH_MAX = 1000

DIFF_COUNT = 6
DIFF_MIN = 1
DIFF_MAX = 5

MODEL_NAME = "3dssd"
DEVICE = 3


CONFIG_IDX_FILE = "/data/3d/mlfs/config_idx.json"


def get_config_idx(widths, diffs):
    """Get configuration index from file.
    {
        idx (int): {
            "widths": [int],
            "diffs": [int],
        }
    }

    Args:
        widths (_type_): _description_
        diffs (_type_): _description_
    """
    if not os.path.exists(CONFIG_IDX_FILE):
        with open(CONFIG_IDX_FILE, "w") as f:
            json.dump({}, f)
    with open(CONFIG_IDX_FILE, "r") as f:
        config_idx = json.load(f)
    for idx, config in config_idx.items():
        if config["widths"] == widths and config["diffs"] == diffs:
            return idx

    config_idx[len(config_idx)] = {"widths": widths, "diffs": diffs}
    with open(CONFIG_IDX_FILE, "w") as f:
        json.dump(config_idx, f)
    return len(config_idx) - 1


def objective(widths, diffs):  # Define an objective function.
    idx = get_config_idx(widths, diffs)
    # Run the sampling C++ program
    print(f"Start sampling: {idx}... ", end="")
    subprocess.run(
        [
            "/workspaces/ml4sys_cirrus/build/combination_sampling",
            "-w",
            ",".join(widths),
            "-d",
            ",".join(diffs),
            "-i",
            str(idx),
        ]
    )
    print("Done")

    # Run inference
    print("Start inference... ", end="")
    with open(f"/data/3d/mlfs/flag/pre_infer/{idx}", "w") as f:
        f.write("1")
    while True:
        if os.path.exists(f"/data/3d/mlfs/flag/post_infer/{idx}"):
            break
    os.remove(f"/data/3d/mlfs/flag/post_infer/{idx}")
    print("Done")

    # Wait for the evaluation result
    with open(f"/data/3d/mlfs/flag/pre_eval/{idx}", "w") as f:
        f.write("1")
    while True:
        if os.path.exists(f"/data/3d/mlfs/flag/post_eval/{idx}"):
            break
    os.remove(f"/data/3d/mlfs/flag/post_eval/{idx}")
    # Read the evaluation result
    with open(f"/data/3d/mlfs/post_eval/{idx}", "r") as f:
        result = json.load(f)
    if result["f1"] >= RAW_F1 - F1_THRESHOLD:
        return -result["saving"] * 100
    else:
        return result["f1"] * 10000 - result["saving"] * 100


def trainable(config):  # Pass a "config" dictionary into your trainable.
    widths = [str(config[f"w{i}"]) for i in range(WIDTH_COUNT)]
    diffs = [str(config[f"d{i}"]) for i in range(DIFF_COUNT)]

    score = objective(widths, diffs)
    train.report({"score": score})  # Send the score to Tune.


def main():
    # Define the search space
    width_space = {
        f"w{i}": tune.uniform(WIDTH_MIN, WIDTH_MAX) for i in range(WIDTH_COUNT)
    }
    diff_space = {f"d{i}": tune.uniform(DIFF_MIN, DIFF_MAX) for i in range(DIFF_COUNT)}
    search_space = {**width_space, **diff_space}

    algo = BayesOptSearch(random_search_steps=4)

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            search_alg=algo,
            num_samples=1,
        ),
        run_config=train.RunConfig(stop={"training_iteration": 40}),
        param_space=search_space,
    )
    results = tuner.fit()

    best_result = results.get_best_result()
    best_config = best_result.config
    print(best_result)
    print(best_config)  # Get the best config


if __name__ == "__main__":
    main()
