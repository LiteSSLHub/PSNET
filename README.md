# Lightweight Contenders: Navigating Semi-Supervised Text Mining through Peer Collaboration and Self Transcendence (PS-NET)

This is the code of Lightweight Contenders: Navigating Semi-Supervised Text Mining through Peer Collaboration and Self Transcendence (PS-NET) with Pytorch.

## Environment Configuration

- numpy
- torch
- transformers
- pyrouge
- rouge
- boto3


Run command below to install all the environment in need(**using python3**)

```shell
pip install -r requirements.txt
```

## Usage

```shell
python src/train.py --task_name ${TASK_NAME}$ \
                    --student_num ${STUDENT_NUM}$ \
                    --config_name_or_path ${CONFIG_NAME_OR_PATH}$ \
                    --teacher_model ${TEACHER_MODEL}$ \
                    --num_training_steps ${NUM_TRAINING_STEPS}$ \
                    --cnndm_dataset_name ${CNNDM_DATASET_NAME}$ \
                    --glue_dataset_name ${GLUE_DATASET_NAME}$ \
                    --train_batch_size ${TRAIN_BATCH_SIZE}$ \
                    --test_batch_size ${TEST_BATCH_SIZE}$ \
                    --val_batch_size ${VAL_BATCH_SIZE}$ \
                    --student_promotion_lr ${STUDENT_PROMOTION_LR}$ \
                    --student_distill_lr ${STUDENT_DISTILL_LR}$ \
                    --student_mutual_lr ${STUDENT_MUTUAL_LR}$ \
                    --teacher_lr ${TEACHER_LR}$ \
                    --lambdau ${LAMBDAU}$ \
                    --rampup_rate ${RAMPUP_RATE}$ \
                    --belittle ${BELITTLE}$ \
```

All the example scripts can be found in `script`
