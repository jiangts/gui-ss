#!/bin/sh
export JOB_DIR=mlflow
export TRAIN_STEPS=1000
export EVAL_STEPS=1
export BATCH_SIZE=16
export DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=mlflow_$DATE
export REGION=us-central1
export GCS_JOB_DIR=gs://ui-scene-seg_training/jobs/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
 --stream-logs \
 --runtime-version 2.1 \
 --python-version 3.5 \
 --job-dir $GCS_JOB_DIR \
 --package-path trainer \
 --module-name trainer.task \
 --region $REGION \
 --scale-tier basic-tpu \
 -- \
 --train-steps $TRAIN_STEPS \
 --batch-size=$BATCH_SIZE \
 --eval-steps $EVAL_STEPS \
 --n_samples=1000
