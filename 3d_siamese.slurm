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
TRAINING_FOLDER=training-list/10
BATCHSIZE=64
EPOCHS=7
STEPS=-1
LEARNING_RATE=0.0000009
LAMBDA=1
MARGIN=40
THRESHOLD=1.3

# create data and algorithm digest
echo "bs=$BATCHSIZE e=$EPOCHS s=$STEPS lr=9e-7 l=$LAMBDA th=$THRESHOLD m=12.0 same_benign batch_norm he_normal\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-tp -1      \
-vp 100      \
-bs $BATCHSIZE      \
-e  $EPOCHS     \
-s  $STEPS        \
-sb           \
-lr $LEARNING_RATE \
-aug         \
-knn        \
-vis        \
-k  1        \
-si 3       \
-l $LAMBDA    \
-th $THRESHOLD      \
-m  $MARGIN     \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"



