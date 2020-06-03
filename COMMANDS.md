https://medium.com/@gogasca_/building-ml-pipelines-for-tensorflow-in-google-cloud-ai-platform-using-mlflow-38ef4c23acf0

Create instance
```bash
export IMAGE_FAMILY="tf-latest-cpu"
export ZONE="us-central1-b"
export INSTANCE_NAME="mlflow-server"
gcloud compute instances create $INSTANCE_NAME \
--zone=$ZONE \
--image-family=$IMAGE_FAMILY \
--machine-type=n1-standard-8 \
--image-project=deeplearning-platform-release \
--maintenance-policy=TERMINATE \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--tags http-server,https-server
```




Create virtual env
```bash
virtualenv -p `which python3` venv
source venv/bin/activate

pip install mlflow tensorflow
pip freeze | grep mlflow
```

Get example code:
```bash
git clone --depth=1 https://github.com/GoogleCloudPlatform/ml-on-gcp.git
cd ml-on-gcp/tutorials/tensorflow/mlflow_gcp/
```

Setup:
```bash
pip install -r requirements.txt
```

Local training:
```bash
export JOB_DIR=mlflow
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=1
export BATCH_SIZE=128

python -m trainer.task \
--train-files=$TRAIN_FILE \
--eval-files=$EVAL_FILE \
--job-dir=$JOB_DIR \
--train-steps=$TRAIN_STEPS \
--eval-steps=$EVAL_STEPS \
--batch-size=$BATCH_SIZE \
--num-epochs=5
```

Confused by section:
> We will log parameters, metrics, save the TensorFlow model in SavedModel format and TensorBoard event information:






Handy commands:
```bash
mlflow ui
```

```bash
export JOB_DIR=mlflow
export TRAIN_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.data.csv
export EVAL_FILE=gs://cloud-samples-data/ml-engine/census/data/adult.test.csv
export TRAIN_STEPS=1000
export EVAL_STEPS=1
export BATCH_SIZE=128

export DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=mlflow_$DATE
export REGION=us-central1
export GCS_JOB_DIR=gs://ui-scene-seg_training/jobs/$JOB_NAME
gcloud ai-platform jobs submit training $JOB_NAME \
 --stream-logs \
 --runtime-version 1.15 \
 --python-version 3.5 \
 --job-dir $GCS_JOB_DIR \
 --package-path trainer \
 --module-name trainer.task \
 --region $REGION \
 -- \
 --train-files $TRAIN_FILE \
 --eval-files $EVAL_FILE \
 --train-steps $TRAIN_STEPS \
 --batch-size=$BATCH_SIZE \
 --eval-steps $EVAL_STEPS \
```


might be a good resource:
https://medium.com/google-cloud-platform-by-cloud-ace/serverless-machine-learning-gcp-3df790270e19




New commands:
Tensorboard
`tensorboard --logdir=gs://ui-scene-seg_training/jobs/`

To run locally
```
. venv/bin/activate
export JOB_DIR=mlflow
export TRAIN_STEPS=100
export EVAL_STEPS=1
export BATCH_SIZE=128

python -m trainer.task \
 --registry_path=/Users/jiangts/Documents/stanford/cs231n/final_project/classify.txt \
 --job-dir=$JOB_DIR \
 --train-steps=$TRAIN_STEPS \
 --eval-steps=$EVAL_STEPS \
 --buffer-size=4 \
 --batch-size=4 \
 --num-epochs=5
```
To run remotely
```
export DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=mlflow_$DATE

export SAMPLES=50000
export BATCH_SIZE=16
export TRAIN_STEPS=1000
export EVAL_STEPS=1
export JOB_DIR=mlflow
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
 --n_samples=$SAMPLES
 ```
