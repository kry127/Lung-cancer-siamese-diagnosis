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
TRAINING_FOLDER=training-list/1

# generate data for CNN training
python3 generate_training.py \
-F $TRAINING_FOLDER \
-t  200      \
-v  30     

# create data and algorithm digest
echo "th=0.8 tp=600 e=50 m=3.5" > ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-tp 600      \
-vp 100      \
-bs 100       \
-e  50       \
-s  1        \
-lr 0.000003 \
-k  7        \
-th 0.8      \
-m  3.5        \
-S "$TRAINING_FOLDER/default.model"



