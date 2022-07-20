#!/bin/bash
dir="server/outputs/"   # you will need to make outputs directory using 'mkdir outputs'
mkdir $dir    # comment this out if directory already made
ram="32G"        # change these

data_dir="/user/work/mc15445/summer-project"
epochs=100

job_name="pose"
time="0-2:00"

task="edge_2d shear real"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="edge_2d shear sim"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="surface_2d shear real"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram
task="surface_2d shear sim"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --dir $data_dir --task $task --ram

# on dev-data
task="edge_2d shear real"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --task $task --ram
task="surface_2d shear real"
sbatch -t $time -J $job_name -o $dir'.out' -e $dir'.err' --mem=$ram server/submit_job.sh python trainer.py --epochs $epochs --task $task --ram
