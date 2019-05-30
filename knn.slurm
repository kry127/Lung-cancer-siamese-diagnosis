#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J 3d-siamese-knn
#SBATCH -o log/3d-siamese-knn-%j.out
#SBATCH -e log/3d-siamese-knn-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation

# specify training folder for containing training and validation links
TRAINING_FOLDER=training-list/6
BATCHSIZE=32
EPOCHS=0
STEPS=0
LEARNING_RATE=0.000006
LAMBDA=8
MARGIN=40
THRESHOLD=15.0

# create data and algorithm digest
echo "bs=$BATCHSIZE e=$EPOCHS s=$STEPS lr=$LEARNING_RATE l=$LAMBDA th=$THRESHOLD m=$MARGIN same_benign\
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
-k  1        \
-si 3       \
-l $LAMBDA    \
-th $THRESHOLD      \
-m  $MARGIN     \
-V "$TRAINING_FOLDER/visual" \
-L "$TRAINING_FOLDER/default.model"



