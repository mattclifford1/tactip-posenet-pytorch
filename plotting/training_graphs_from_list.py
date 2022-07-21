import matplotlib.pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser

'''run with eg:
$ python plotting/training_graphs_from_list.py --dir ~/Downloads/pose_estimation/
'''

if __name__ == '__main__':
    parser = ArgumentParser(description='Plot training graphs')
    parser.add_argument("--dir", default=os.path.join(os.path.expanduser('~'), 'Downloads', 'pose_estimation'), help='path to folder where training graphs are within')
    ARGS = parser.parse_args()

    # define the  label:filepath   to plot
    train_routine = '_LR:0.0001_BS:16'
    curves_to_plot = {
        # 'edge tap real': os.path.join('edge_2d', 'tap', 'real'+train_routine, 'run_0', 'training_stats.csv'),
        # 'edge tap sim': os.path.join('edge_2d', 'tap', 'sim'+train_routine, 'run_0', 'training_stats.csv'),
        # 'edge shear real': os.path.join('edge_2d', 'shear', 'real'+train_routine, 'run_0', 'training_stats.csv'),
        # 'edge shear sim': os.path.join('edge_2d', 'shear', 'sim'+train_routine, 'run_0', 'training_stats.csv'),

        'surface tap real': os.path.join('surface_3d', 'tap', 'real'+train_routine, 'run_1', 'training_stats.csv'),
        'surface tap sim': os.path.join('surface_3d', 'tap', 'sim'+train_routine, 'run_1', 'training_stats.csv'),
        'surface shear real': os.path.join('surface_3d', 'shear', 'real'+train_routine, 'run_1', 'training_stats.csv'),
        'surface shear sim': os.path.join('surface_3d', 'shear', 'sim'+train_routine, 'run_1', 'training_stats.csv'),

        # 'edge shear nathan': os.path.join('edge_2d', 'shear', 'nathan'+train_routine, 'run_0', 'training_stats.csv'),
        # 'surface shear nathan': os.path.join('surface_2d', 'shear', 'nathan'+train_routine, 'run_0', 'training_stats.csv'),
    }

    cols = ['mean training loss', 'val MAE']
    fig, ax = plt.subplots(nrows=1, ncols=len(cols), figsize=(17,11))

    for i, col in enumerate(ax):
        for key in curves_to_plot.keys():
            file = curves_to_plot[key]
            df = pd.read_csv(os.path.join(ARGS.dir, file))
            # print(df['epoch'].values)
            col.plot(df['epoch'].values[1:], df[cols[i]].values[1:], label=key)
            # if i == 0:
            #     col.set_ylabel('epoch')
            col.set_xlabel('epoch')

        col.legend()
        col.set_title(cols[i])
    plt.show()
