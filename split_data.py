'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

code adapted from nathan lepora's tactip jupyter notebook
'''
import os, json, pandas as pd, numpy as np
from argparse import ArgumentParser
from pathlib import Path, PureWindowsPath

def make_split(ARGS):
    # Paths and files
    data_folder = os.path.join(ARGS.data_dir, ARGS.sensor_type, ARGS.model_type)
    model_folder = os.path.join(ARGS.sensor_type, ARGS.model_type)
    meta_filepath = os.path.join(data_folder, "meta.json")
    targets_filepath = os.path.join(data_folder, "targets.csv")

    # Create new dataset
    for set in ["train", "test"]:
        set_folder = os.path.join(data_folder, set)
        os.makedirs(set_folder, exist_ok=True)

        # Load meta data and targets
        with open(meta_filepath, 'r') as f:
            meta = json.load(f)
        target_df = pd.read_csv(targets_filepath)

        # Select data
        np.random.seed(0) # make predictable
        inds = np.random.choice([True, False], size=(meta["num_poses"]), p=[ARGS.split, 1-ARGS.split])
        if set=="test":
            inds = np.invert(inds)

        # Save new dataset
        meta["meta_file"] =  os.path.join(model_folder, set, "meta.json")
        meta["target_df_file"] = os.path.join(model_folder, set, "targets.csv")
        meta["num_poses"] = len(target_df[inds])

        # Convert windows specific filepaths to be OS-independant
        for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
            meta[key] = Path(PureWindowsPath(meta[key])).as_posix() # extend compatablility with Unix file systems

        # Save meta data
        with open(os.path.join(ARGS.data_dir, meta["meta_file"]), 'w') as f:
            json.dump(meta, f)
        target_df[inds].to_csv(os.path.join(ARGS.data_dir, meta["target_df_file"]), index=False)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--data_dir", default='dev-data', type=str, help='data base dir')
    parser.add_argument("--sensor_type", default='tactip-127', type=str, help='tactip sensor model type - eg. tactip-127')
    parser.add_argument("--model_type", default='model_surface2d', type=str, help='model type - edge or surface')
    parser.add_argument("--split", default=0.75, type=float, help='train test split proportion')
    ARGS = parser.parse_args()
    print('Running test train split with Arguments:\n', ARGS)
    make_split(ARGS)
