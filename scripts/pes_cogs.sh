# train on vanilla cogs
sbatch train_a100.slurm "--T 10000 --K 50 --model_name EleutherAI/pythia-70m --data_path  \
--train_path ./data/full/train.json --val_path ./data/base/dev.json \
--output_dir ./output \
--bsz 32 --seq_length 256 --k 3 --warmup_steps 50 --seed 0 --lr 5e-5 \
--run_type train '[0,1,2]'"


# meta_train
sbatch train_a100.slurm "--T 10000 --K 50 --max_steps 2000 --model_name EleutherAI/pythia-70m --data_path  \
--train_path ./data/full/train.json --val_path ./data/base/dev.json \
--output_dir ./output \
--bsz 32 --seq_length 256 --k 3 --warmup_steps 50 --seed 0 --lr 5e-5 \
--num_particles 4 --meta_lr 3e-3 --sigma 0.01 --input_dim 3 output_dim 3 --run_type meta_train --skills '[0,1,2]'"
