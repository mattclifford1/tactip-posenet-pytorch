'''
Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk

code adapted from nathan lepora's tactip jupyter notebook
'''
import os, json, pandas as pd, matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tensorflow").addHandler(logging.NullHandler(logging.ERROR))
from pose_models_2d.lib.models.cnn_model import CNNmodel
from argparse import ArgumentParser


def plot_pred(data_path, pred_df, target_names, model_file, meta_file, poses_rng, **kwargs):
    plt.rcParams.update({'font.size': 18})
    n = len(target_names)

    fig, axes = plt.subplots(ncols=n, figsize=(7*n, 7))
    fig.suptitle(model_file.replace(data_path,'') + '\n' +
                 os.path.dirname(meta_file.replace(data_path,'')))
    fig.subplots_adjust(wspace=0.3)
    n_smooth = int(pred_df.shape[0]/20)
    for i, ax in enumerate(axes):
        sort_df = pred_df.sort_values(by=[f"target_{i+1}"])
        ax.scatter(sort_df[f"target_{i+1}"], sort_df[f"pred_{i+1}"], s=1, c=sort_df["target_1"], cmap="inferno")
        ax.plot(sort_df[f"target_{i+1}"].rolling(n_smooth).mean(), sort_df[f"pred_{i+1}"].rolling(n_smooth).mean(), c="red")
        ax.set(xlabel=f"target {target_names[i]}", ylabel=f"predicted {target_names[i]}")
        ind = int(target_names[i][-1])-1
        ax.set_xlim(poses_rng[0][ind], poses_rng[1][ind])
        ax.set_ylim(poses_rng[0][ind], poses_rng[1][ind])
        ax.text(0.05,0.9, 'MAE='+str(sort_df[f"error_{i+1}"].mean())[0:4], transform=ax.transAxes)
        ax.grid(True)
    return fig

def main(ARGS):
    # User-defined paths
    data_folder = os.path.join(ARGS.data_dir, ARGS.sensor_type, ARGS.model_type)
    valid_dir =  os.path.join(data_folder, "test")
    model_abs_dir = os.path.join(data_folder, 'train', "train2d_cnn")
    test_results_rel_dir = os.path.join(ARGS.sensor_type, ARGS.model_type, 'test', "test2d_cnn")
    test_results_abs_dir = os.path.join(data_folder, 'test', "test2d_cnn")

    # Open saved meta dictionaries
    with open(os.path.join(model_abs_dir, "meta.json"), 'r') as f:
        model_meta = json.load(f)
    with open(os.path.join(valid_dir, "meta.json"), 'r') as f:
        valid_meta = json.load(f)


    # Make the new meta dictionary
    meta = {**model_meta,
        # ~~~~~~~~~ Paths ~~~~~~~~~#
        "meta_file": os.path.join(test_results_rel_dir, "meta.json"),
        "test_image_dir": valid_meta["image_dir"],
        "test_df_file": valid_meta["target_df_file"],
        # ~~~~~~~~~ Comments ~~~~~~~~~#
        "comments": "test on validation data"
        }

    # Save dictionary to file
    os.makedirs(test_results_abs_dir, exist_ok=True)
    with open(os.path.join(ARGS.data_dir, meta["meta_file"]), 'w') as f:
        json.dump(meta, f)

    # Absolute paths
    for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
        meta[key] = os.path.join(ARGS.data_dir, meta[key])

    # Startup/load model and make predictions on test data
    cnn = CNNmodel()
    cnn.load_model(**meta)
    pred = cnn.predict_from_file(**meta)

    # Analyze and plot predictions
    pred_df = pd.read_csv(meta["test_df_file"])
    MAEs = []
    for i, item in enumerate(meta["target_names"], start=1):
        pred_df[f"pred_{i}"] = pred[:, i-1]
        pred_df[f"target_{i}"] = pred_df[item]
        pred_df[f"error_{i}"] = abs(pred_df[f"pred_{i}"] - pred_df[f"target_{i}"])
        MAEs.append(pred_df[f"error_{i}"].mean())
    pred_df.to_csv(os.path.join(test_results_abs_dir, "predictions.csv"))
    fig = plot_pred(ARGS.data_dir, pred_df, **meta)
    fig.savefig(os.path.join(test_results_abs_dir, "errors.png"), bbox_inches='tight')
    return MAEs

if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--data_dir", default='dev-data', type=str, help='data base dir')
    parser.add_argument("--sensor_type", default='tactip-127', type=str, help='tactip sensor model type - eg. tactip-127')
    parser.add_argument("--model_type", default='model_surface2d', type=str, help='model type - model_edge2d or model_surface2d')
    ARGS = parser.parse_args()
    print('Running test train split with Arguments:\n', ARGS)
    MAEs = main(ARGS)
    print('MAEs: ', MAEs)
