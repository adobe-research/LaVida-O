export WANDB_DIR=/mnt/localssd/wandb
set -x
aws configure set default.s3.max_concurrent_requests 2

LLADA_8B_INSTRUCT=/mnt/localssd/lavida-llada-v1.0-instruct-gen-converted-v3-2048
VISION_MODEL_VERSION="/mnt/localssd/siglip-so400m-patch14-384"

LLM_VERSION=$LLADA_8B_INSTRUCT

LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"


VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH=config/data/s2_t2i_pretrain.yaml
IMG_PATH=/mnt/localssd/


export ALWASY_DO_2DPOOL=1 

PROMPT_VERSION="qwen_1_5"
PROMPT_VERSION="llada"
RUN_NAME=lavida-o-pretrain
MID_RUN_NAME=$RUN_NAME
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
export NUM_GPUS=${NUM_GPUS:-"8"}
PORT=23334
export SELECT_ONE_INDEX=1
export DEBUG_FIX_PADDING=1
BASE_RUN_NAME=/path/to/projectors
export SKIP_COMPLEMENTARY_MASKING=1
export IGNORE_TEXT_LOSS=1
export SKIP_DOWN_SAMPLE=1

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
torchrun \
    --nnodes=${WORLD_SIZE} \
    --node_rank ${RANK} \
    --nproc_per_node="${NUM_GPUS}"  \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path $DATA_PATH \
    --load_vlm True \
    --image_folder $IMG_PATH \
    --mm_tunable_parts="mm_language_model_vision_parms" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/mnt/localssd/outputdir/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --resume_from_checkpoint latest \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --vqvae /mnt/localssd/Meissonic/vqvae \
    --unified_gen \
    --dual_tower \
    --prompt_drop_rate 0.1 \
    --image_gen_size 256 \
    --dual_tower_layers 16 \
    --num_gen_image_tokens 256 \
    --t2i_eval True  ${@}  \
    --num_train_epochs 10 \
    --max_steps 300000 \