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
BATCHSIZE=128
EPOCHS=1
LAMBDA=0.0818
THRESHOLD=1.0

# create data and algorithm digest
echo "bs=$BATCHSIZE e=$EPOCHS  lr=9e-7 l=$LAMBDA th=$THRESHOLD m=12.0 batch_norm he_normal\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-tp -1      \
-vp 100      \
-bs $BATCHSIZE      \
-e  $EPOCHS     \
-s  -1        \
-lr 0.0000009 \
-aug         \
-k  7        \
-si 1       \
-l $LAMBDA    \
-th $THRESHOLD      \
-m  12.0        \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"



