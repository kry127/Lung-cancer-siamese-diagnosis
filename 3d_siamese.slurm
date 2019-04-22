#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J 3d-siamese-training 
#SBATCH -o log/3d-siamese-training-%j.out
#SBATCH -e log/3d-siamese-training-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation

srun python3 3d_siamese.py \
-t  70      \
-v  30      \
-tp 400     \
-vp 100     \
-bs 100     \
-e  70     \
-s  3       \
-k  7       \
-th 1       \
-m  3       \
-S ./big_learn_set.model



