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
EPOCHS=3
STEPS=10
LEARNING_RATE=0.0000018
LEARNING_RATE_REDUCE_FACTOR_KEY="-rf 0.99"
MARGIN=30
LAMBDA1=1
LAMBDA2=12
LAMBDA3=0.0000000001
LAMBDA4=0.0000000001
LAMBDA5=0.0000000001
LAMBDA6=0.0000000001
LAMBDAS="$LAMBDA1;$LAMBDA2;$LAMBDA3;$LAMBDA4;$LAMBDA5;$LAMBDA6"

# create data and algorithm digest
echo "bs=$BATCHSIZE e=$EPOCHS s=$STEPS lr=$LEARNING_RATE l=$LAMBDAS m=$MARGIN same_benign\
" >> ./$TRAINING_FOLDER/digest.txt

# run HPC task
srun python3 3d_siamese.py \
-F $TRAINING_FOLDER \
-bs $BATCHSIZE      \
-e  $EPOCHS     \
-s  $STEPS        \
-sb           \
-lr $LEARNING_RATE \
$LEARNING_RATE_REDUCE_FACTOR_KEY \
-aug         \
-vis        \
-k  1        \
-si 3       \
-th 15.0       \
-m  $MARGIN     \
-l1 $LAMBDA1    \
-l2 $LAMBDA2    \
-l3 $LAMBDA3    \
-l4 $LAMBDA4    \
-l5 $LAMBDA5    \
-l6 $LAMBDA6    \
-V "$TRAINING_FOLDER/visual" \
-L "$TRAINING_FOLDER/default.model" \
-S "$TRAINING_FOLDER/default.model"



