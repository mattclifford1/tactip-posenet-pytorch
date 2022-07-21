#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
epochs=100

time="0-1:20"

# sim2real data
task="edge_2d shear real"
name="edge_real"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="edge_2d shear sim"
name="edge_sim"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="surface_3d shear real"
name="surface_real"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="surface_3d shear sim"
name="surface_sim"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram

# # on nathans data
data_dir="/user/work/mc15445/summer-project/data/Nathan/tactip-127"
task="edge_2d shear real"
name="edge_nathan"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="surface_2d shear real"
name="surface_nathan"
sbatch -t $time -J $name -o $dir$name'.out' -e $dir$name'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
