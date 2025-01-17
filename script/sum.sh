export TOKENIZERS_PARALLELISM=false

gpu="0"

student_num="2"
lambdau="1"
rampup_rate="0.2"
belittle="0.8"

task_name="sum"
dataset="--cnndm_dataset_name cnndm --glue_dataset_name yahoo"
batch_size="--train_batch_size 4 --test_batch_size 32 --val_batch_size 32"
student_promotion_lr="2e-3"
student_distill_lr="5e-5"
student_mutual_lr="1e-4"
teacher_lr="2e-3"

config_name_or_path="config/2layer_890m"

teacher_model="/home/LAB/liujn/jwf/NewConsistSum/tmp/newbert/model_0/"
tokenizer_name_or_path="/home/LAB/liujn/jwf/NewConsistSum/tmp/newbert/model_0/"

num_training_steps="10000"

root_dir="experiments/cnndm/cluster/"
ckpt_dir="sum"
mkdir -p $root_dir


python src/train.py --task_name $task_name $dataset $batch_size --num_workers 8 --root_dir $root_dir --ckpt_dir $ckpt_dir --student_mutual_lr $student_mutual_lr --student_distill_lr $student_distill_lr\
        --num_training_steps $num_training_steps --teacher_model $teacher_model --student_promotion_lr $student_promotion_lr --teacher_lr $teacher_lr --tokenizer_name_or_path $tokenizer_name_or_path\
        --val_interval 100 --config_name_or_path $config_name_or_path --student_num $student_num --lambdau $lambdau --rampup_rate $rampup_rate --belittle $belittle \
        --cuda --gpu $gpu --do_adversary_teacher --do_adversary_1 --do_adversary_3 --do_cluster --supervised_size 100 --unsupervised_size 5000 --distill_mode 0
