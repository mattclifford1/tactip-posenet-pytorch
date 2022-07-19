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
from argparse import ArgumentParser
import sys; sys.path.append('..'); sys.path.append('.')
from keras_version.cnn_model import CNNmodel


def plot_history(history_df):
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(ncols=1, figsize=(7, 7))
    history_df.plot(ax=ax, y="loss", label="Training loss")
    history_df.plot(ax=ax, y="val_loss", label="Validation loss")
    ax.set(xlabel="epochs", ylabel="loss"); ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.suptitle(f"Training loss: {min(history_df.loss):.2f} Validation loss: {min(history_df.val_loss):.2f}")
    return fig


def main(ARGS):
    # Paths and files
    data_folder = os.path.join(ARGS.data_dir, ARGS.sensor_type, ARGS.model_type)
    train_dir =  os.path.join(data_folder, "train")
    valid_dir =  os.path.join(data_folder, "test")
    model_rel_dir = os.path.join(ARGS.sensor_type, ARGS.model_type, 'train', "train2d_cnn")
    model_abs_dir = os.path.join(data_folder, 'train', "train2d_cnn")

    # Open saved meta dictionaries
    with open(os.path.join(train_dir, "meta.json"), 'r') as f:
        train_meta = json.load(f)
    with open(os.path.join(valid_dir, "meta.json"), 'r') as f:
        valid_meta = json.load(f)

    # Make the new meta dictionary
    meta = {**train_meta,
        # ~~~~~~~~~ Paths ~~~~~~~~~#
        "meta_file": os.path.join(model_rel_dir, "meta.json"),
        "model_file": os.path.join(model_rel_dir, "model.h5"),
        "train_image_dir": train_meta["image_dir"],
        "valid_image_dir": valid_meta["image_dir"],
        "train_df_file": train_meta["target_df_file"],
        "valid_df_file": valid_meta["target_df_file"],
        # ~~~~~~~~~ Model parameters ~~~~~~~~~#
        "num_conv_layers": 5,
        "num_conv_filters": 256,
        "num_dense_layers": 1,
        "num_dense_units": 64,
        "activation": 'elu',
        "dropout": 0.06,
        "kernel_l1": 0.0006,
        "kernel_l2": 0.001,
        "batch_size": 16,
        "epochs": ARGS.epochs,
        "patience": 10,
        "lr": 1e-4,
        "decay": 1e-6,
        "target_names": ["pose_2", "pose_6"],
        # ~~~~~~~~~ Camera settings ~~~~~~~~~#
        "size": [128, 128], # TacTip
    #     "size": [160, 120], # DigiTac/Digit
        # ~~~~~~~~~ Comments ~~~~~~~~~#
        "comments": "training 2d cnn model"
        }

    # Save dictionary to file
    os.makedirs(model_abs_dir, exist_ok=True)
    with open(os.path.join(ARGS.data_dir, meta["meta_file"]), 'w') as f:
        json.dump(meta, f)

    # Absolute paths
    for key in [k for k in meta.keys() if "file" in k or "dir" in k]:
        meta[key] = os.path.join(ARGS.data_dir, meta[key])

    cnn = CNNmodel()
    cnn.build_model(**meta)
    cnn.print_model_summary()
    # history = cnn.fit_model(**meta, verbose=1)
    #
    # # Save and plot training history
    # history_df = pd.DataFrame(history)
    # history_df.index += 1
    # history_df.to_csv(os.path.join(model_abs_dir, "history.csv"), index=False)
    # fig = plot_history(history_df)
    # fig.savefig(os.path.join(model_abs_dir, "history.png"), bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    # get command line arguments
    parser = ArgumentParser(description='data dir and model type')
    parser.add_argument("--data_dir", default='dev-data', type=str, help='data base dir')
    parser.add_argument("--sensor_type", default='tactip-127', type=str, help='tactip sensor model type - eg. tactip-127')
    parser.add_argument("--model_type", default='model_surface2d', type=str, help='model type - model_edge2d or model_surface2d')
    parser.add_argument("--epochs", default=2, type=int, help='how many epochs to train for')
    ARGS = parser.parse_args()
    print('Running test train split with Arguments:\n', ARGS)
    main(ARGS)
