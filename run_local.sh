#!/bin/sh
. venv/bin/activate
export JOB_DIR=mlflow
export TRAIN_STEPS=100
export EVAL_STEPS=1
export BATCH_SIZE=128

python -m trainer.task --images_path=/Users/jiangts/Documents/stanford/cs231n/final_project/combined --annotations_path=/Users/jiangts/Documents/stanford/cs231n/final_project/semantic_annotations --job-dir=$JOB_DIR --train-steps=$TRAIN_STEPS --eval-steps=$EVAL_STEPS --batch-size=4 --num-epochs=5 --n_samples=128
