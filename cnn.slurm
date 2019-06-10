#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p tornado-k40
#SBATCH -t 10-00:00:00
#SBATCH -J cnn-training 
#SBATCH -o log/cnn-training-%j.out
#SBATCH -e log/cnn-training-%j.err

if [ -f /etc/profile.d/modules-basis.sh ]; then
    source /etc/profile.d/modules-basis.sh 
fi

module purge
source ~/activation

# specify training folder for containing training and validation links
TRAINING_FOLDER=training-list/11
BATCHSIZE=32
EPOCHS=25
STEPS=-1
LEARNING_RATE=0.00077

# create data and algorithm digest
echo "cnn resnet bs=$BATCHSIZE e=$EPOCHS s=$STEPS lr=$LEARNING_RATE\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 cnn.py \
-F $TRAINING_FOLDER \
-bs $BATCHSIZE      \
-e  $EPOCHS     \
-s  $STEPS        \
-lr $LEARNING_RATE \
-aug         \
-vis        \
-V "$TRAINING_FOLDER/visual" \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"