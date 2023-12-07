
MODEL_PATH='/home/vislab-001/Jared/Envy_AI_City/slowfast/checkpoints/vit_b_k710_dl_from_giant.pth'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

DATA_PATH="/home/vislab-001/Jared/Envy_AI_City/slowfast/annotations/Dashboard/fold_0" # The data list folder. the folder has three files: train.csv, val.csv, test.csv
OUTPUT_DIR="/home/vislab-001/Jared/Envy_AI_City/VideoMAEv2/training_output_dash/fold_0" # Your output folder for deepspeed config file, logs and checkpoints
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 12320 --nnodes=1 --node_rank=0 --master_addr=localhost \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set AI-City-Track-3 \
        --nb_classes 16 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 200 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \


DATA_PATH="/home/vislab-001/Jared/Envy_AI_City/slowfast/annotations/Rear_view/fold_0" # The data list folder. the folder has three files: train.csv, val.csv, test.csv
OUTPUT_DIR="/home/vislab-001/Jared/Envy_AI_City/VideoMAEv2/training_output_rear/fold_0" # Your output folder for deepspeed config file, logs and checkpoints
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 12320 --nnodes=1 --node_rank=0 --master_addr=localhost \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set AI-City-Track-3 \
        --nb_classes 16 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 200 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \


DATA_PATH="/home/vislab-001/Jared/Envy_AI_City/slowfast/annotations/Right_side_window/fold_0" # The data list folder. the folder has three files: train.csv, val.csv, test.csv
OUTPUT_DIR="/home/vislab-001/Jared/Envy_AI_City/VideoMAEv2/training_output_right/fold_0" # Your output folder for deepspeed config file, logs and checkpoints
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 12320 --nnodes=1 --node_rank=0 --master_addr=localhost \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set AI-City-Track-3 \
        --nb_classes 16 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 200 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.75 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed 