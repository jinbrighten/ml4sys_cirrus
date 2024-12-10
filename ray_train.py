import json
import os
import subprocess
from pathlib import Path

from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB

from inference import inference_directory

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

NAME = "bohb"

RAW_F1 = 0.73
F1_THRESHOLD = 0.08

WIDTH_COUNT = 6
WIDTH_MIN = 10
WIDTH_MAX = 1000

DIFF_COUNT = 6
DIFF_MIN = 0
DIFF_MAX = 5

MODEL_NAME = "3dssd"
DEVICE = 3


MLFS_ROOT = Path("/data/3d/mlfs/") / NAME
# CONFIG_IDX_FILE = "/data/3d/mlfs/config_idx.json"
CONFIG_IDX_FILE = MLFS_ROOT / "config_idx.json"


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
            "-p",
            str(MLFS_ROOT / "pre_infer") + "/",
        ]
    )
    print("Done")

    # Run inference
    print("Start inference... ", end="")
    with open(MLFS_ROOT / f"flag/pre_infer/{idx}", "w") as f:
        f.write("1")
    while True:
        if os.path.exists(MLFS_ROOT / f"flag/post_infer/{idx}"):
            break
    os.remove(MLFS_ROOT / f"flag/post_infer/{idx}")
    print("Done")

    # Wait for the evaluation result
    with open(MLFS_ROOT / f"flag/pre_eval/{idx}", "w") as f:
        f.write("1")
    while True:
        if os.path.exists(MLFS_ROOT / f"flag/post_eval/{idx}"):
            break
    os.remove(MLFS_ROOT / f"flag/post_eval/{idx}")
    # Read the evaluation result
    with open(MLFS_ROOT / f"post_eval/{idx}", "r") as f:
        result = json.load(f)
    if result["f1"] >= RAW_F1 - F1_THRESHOLD:
        return -result["saving"] * 100
    else:
        return result["f1"] * 10000 - result["saving"] * 100


def trainable(config):  # Pass a "config" dictionary into your trainable.
    widths = [str(int(config[f"w{i}"])) for i in range(WIDTH_COUNT)]
    diffs = [str(int(config[f"d{i}"])) for i in range(DIFF_COUNT)]

    score = objective(widths, diffs)
    train.report({"score": score})  # Send the score to Tune.


def main():
    # Define the search space
    width_space = {
        f"w{i}": tune.uniform(WIDTH_MIN, WIDTH_MAX) for i in range(WIDTH_COUNT)
    }
    diff_space = {f"d{i}": tune.uniform(DIFF_MIN, DIFF_MAX) for i in range(DIFF_COUNT)}
    search_space = {"iterations": 100, **width_space, **diff_space}

    MLFS_ROOT.mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "flag/post_eval/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "flag/post_infer/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "flag/pre_eval/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "flag/pre_infer/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "post_eval/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "post_infer/").mkdir(exist_ok=True, parents=True)
    (MLFS_ROOT / "pre_infer/").mkdir(exist_ok=True, parents=True)

    ### BOHB
    if NAME == "bohb":
        # config = {
        #     "iterations": 100,
        #     "width": tune.uniform(0, 20),
        #     "height": tune.uniform(-100, 100),
        #     "activation": tune.choice(["relu", "tanh"]),
        # }

        # Optional: Pass the parameter space yourself
        # import ConfigSpace as CS
        # config_space = CS.ConfigurationSpace()
        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter("width", lower=0, upper=20))
        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter("height", lower=-100, upper=100))
        # config_space.add_hyperparameter(
        #     CS.CategoricalHyperparameter(
        #         "activation", choices=["relu", "tanh"]))

        max_iterations = 3200
        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=max_iterations,
            reduction_factor=2,
            stop_last_trials=False,
        )

        bohb_search = TuneBOHB(
            # space=config_space,  # If you want to set the space manually
            points_to_evaluate=[
                {
                    "d0": 4.99091870401381,
                    "d1": 2.4945489584650207,
                    "d2": 1.6186965958978723,
                    "d3": 1.8720283375993811,
                    "d4": 4.388774874685426,
                    "d5": 2.2231877877633024,
                    "w0": 910.0336144087651,
                    "w1": 459.5825933227906,
                    "w2": 69.46064018393379,
                    "w3": 671.9777106716047,
                    "w4": 613.6999622523164,
                    "w5": 950.1263139445899,
                },
            ]
        )
        bohb_search = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

        tuner = tune.Tuner(
            trainable,
            run_config=train.RunConfig(
                name="bohb_test", stop={"training_iteration": max_iterations}
            ),
            tune_config=tune.TuneConfig(
                metric="score",
                mode="min",
                scheduler=bohb_hyperband,
                search_alg=bohb_search,
                num_samples=32,
            ),
            param_space=search_space,
        )

    ### Bayes
    elif NAME == "bayes":
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
