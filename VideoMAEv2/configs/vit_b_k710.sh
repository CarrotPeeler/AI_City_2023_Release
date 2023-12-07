# finetune data list file follows the following format
# for the video data line: video_path, label
# for the rawframe data line: frame_folder_path, total_frames, label

OUTPUT_DIR='/home/vislab-001/Jared/Envy_AI_City/VideoMAEv2/training_output' # Your output folder for deepspeed config file, logs and checkpoints
DATA_PATH='/home/vislab-001/Jared/Envy_AI_City/slowfast' # The data list folder. the folder has three files: train.csv, val.csv, test.csv
MODEL_PATH='/home/vislab-001/Jared/Envy_AI_City/slowfast/checkpoints/vit_b_k710_dl_from_giant.pth'
IMG_DIFF_JSON_PATH='/home/vislab-001/Jared/Envy_AI_City/data/img_diff_aicity_train.json'

JOB_NAME=$1
PARTITION=${PARTITION:-"video"}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port 12320 --nnodes=1 --node_rank=0 --master_addr=localhost \
        run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_set AI-City-Track-3 \
        --img_diff_json_path ${IMG_DIFF_JSON_PATH} \
        --sample_mode normal \
        --nb_classes 16 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 6 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 20 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 8 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.1 \
        --head_drop_rate 0.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --warmup_epochs 5 \
        --epochs 200 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \