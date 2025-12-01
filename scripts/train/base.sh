# replace with 256 checkpoint
LLADA_8B_INSTRUCT=/mnt/localssd/lavida-llada-v1.0-instruct-gen-converted-v3-2048
VISION_MODEL_VERSION="/mnt/localssd/siglip-so400m-patch14-384"

LLM_VERSION=$LLADA_8B_INSTRUCT
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

DATA_PATH=config/data/debug_dataset.yaml
IMG_PATH=/mnt/localssd/
RUN_NAME=sft-t2i
export WANDB_DIR=/mnt/localssd/wandb


export ALWASY_DO_2DPOOL=1 

PROMPT_VERSION="qwen_1_5"
PROMPT_VERSION="llada"


MID_RUN_NAME=$RUN_NAME
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION 
NUM_GPUS=${NUM_GPUS:-"8"}
NUM_CKPT=${NUM_CKPT:-"64"}
BATCH_SIZE=${BATCH_SIZE:-"4"}
# PORT=23334
export SELECT_ONE_INDEX=0
export DEBUG_FIX_PADDING=1
export SELECT_ALL_INDEX=1
export NOT_ALWASY_DO_2DPOOL=1
export SKIP_COMPLEMENTARY_MASKING=1
set -x
BASE_RUN_NAME=/path/to/projectors

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
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_vision_resampler,mm_language_model,mm_language_model_vision_parms" \
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
    --mm_spatial_pool_mode conv \
    --mm_spatial_pool_stride 2 \
    --mm_resampler_type spatial_pool \
    --mm_spatial_pool_out_channels 1152 \
    --add_loc_tokens \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/mnt/localssd/outputdir/${MID_RUN_NAME}" \
    --num_train_epochs 10 \
    --max_steps 100000  \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --resume_from_checkpoint latest \
    --lr_scheduler_kwargs '{"min_lr_rate":0.1}' \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --lmms_eval_generate_tasks=vqav2_val_lite,chartqa_lite,textvqa_val_lite,docvqa_val_lite,infovqa_val_lite \
    --lmms_eval_extra_tasks "refcoco" \
    --vqvae /mnt/localssd/Meissonic/vqvae \
    --unified_gen \
    --dual_tower ${@} \
    --prompt_drop_rate 0.1 \
    --image_gen_size 1024 \
    --dual_tower_layers 16 \
    --num_gen_image_tokens 1024 \
    --t2i_eval True \
    --policy cosine \
    ${@} \


