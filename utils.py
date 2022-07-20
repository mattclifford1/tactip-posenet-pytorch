'''
Train helper functions

Author: Matt Clifford
Email: matt.clifford@bristol.ac.uk
'''
import os
import shutil
import torch
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   # torch warning we dont care about

def check_task(task_tuple):
    task0 = ['surface_3d', 'edge_2d']
    task1 = ['tap', 'shear']
    if len(task_tuple) != 2:
        raise Exception('Task needs to be length 2')
    if task_tuple[0] not in task0:
        raise Exception('first task arg needs to be either: ', task0, ' not:', str(task_tuple[0]))
    if task_tuple[1] not in task1:
        raise Exception('second task arg needs to be either: ', task1, ' not:', str(task_tuple[1]))

class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class train_saver:
    def __init__(self, base_dir,
                       model,
                       lr,
                       batch_size,
                       task,
                       save_name=''):
        self.base_dir = base_dir
        self.task = task
        self.save_name = save_name
        # if hasattr(model, 'dimensions'):
        #     self.model_dimensions = model.dimensions
        # else:
        #     self.model_dimensions = ''
        self.lr = lr
        self.batch_size = batch_size
        self.get_save_dir()

    def get_save_dir(self):
        dir = os.path.join(self.base_dir, self.task)
        name = self.save_name
        name += 'LR:'+str(self.lr)
        name += '_BS:'+str(self.batch_size)
        self.dir = os.path.join(dir, name)
        # find if there are previous runs
        run_name = 'run_'
        if os.path.isdir(self.dir):
            # shutil.rmtree(self.dir)
            runs = [int(i[len(run_name):]) for i in os.listdir(self.dir)]
            run_num = max(runs) + 1
        else:
            run_num = 0
        self.dir = os.path.join(self.dir, run_name+str(run_num))
        self.models_dir = os.path.join(self.dir, 'checkpoints')
        # make dirs is dont already exist
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_pretrained(self, model):
        checkpoints = os.listdir(self.models_dir)
        saves = []
        for checkpoint in checkpoints:
            name, ext = os.path.splitext(checkpoint)
            try:
                epoch = int(name)
                saves.append(epoch)
            except:
                pass
        if len(saves) > 0:
            latest_epoch = max(saves)
            weights_path = os.path.join(self.models_dir, str(latest_epoch)+'.pth')
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            print('Loaded previously trained model at epoch: '+str(latest_epoch))
            return latest_epoch
        else:
            return 0 #no pretrained found

    def save_model(self, model, name):
        # print(os.path.join(self.models_dir, str(name)+'.pth'))
        torch.save(model.state_dict(), os.path.join(self.models_dir, str(name)+'.pth'))

    def log_training_stats(self, stats_dicts):
        df = pd.DataFrame(stats_dicts)
        # load csv if there is one
        file = os.path.join(self.dir, 'training_stats.csv')
        if os.path.isfile(file):
            df.to_csv(file, mode='a', index=False, header=False)
        else:
            df.to_csv(file, index=False)
