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

# specify training folder for containing training and validation links
TRAINING_FOLDER=training-list/6

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-tp 0      \
-vp 0      \
-bs 2      \
-e  0       \
-s  0        \
-lr 0.0000009 \
-k  9        \
-si 4.0     \
-l 1.0    \
-th 2.0      \
-m  15.0        \
-knn              \
-L "$TRAINING_FOLDER/default.model"



