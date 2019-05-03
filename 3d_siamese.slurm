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
TRAINING_FOLDER=training-list/4

# create data and algorithm digest
echo "tp=128 e=20 s=3 lr=6e-5 l=6e-4 th=1.0 m=10000.0 batch normalization\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-tp 128      \
-vp 100      \
-bs 32      \
-e  20       \
-s  3        \
-lr 0.00006 \
-k  7        \
-l 0.0006    \
-th 1.0      \
-m  10000.0        \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"



