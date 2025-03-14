export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_aircraft_4_shot.txt"

export DATA_PATH=./view_classification_data
export CKPT_PATH=Qwen/Qwen2-VL-2B-Instruct
export SAVE_PATH=./share_models/Qwen2-VL-2B-R1-Echo-View-Classification


torchrun --nproc_per_node="7" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/virft/src/open_r1/grpo_classification.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed ./zero3.json \
    --max_prompt_length 2056 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
     --fp16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --max_pixels 401408 \
    --num_train_epochs 4 \
    --run_name view_classification_r1_num_gen_2 \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2
