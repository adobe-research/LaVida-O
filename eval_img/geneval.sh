

ANNOTATION='eval_img/evaluation_metadata.jsonl'
BASE_MODEL_PATH=none
CKPT=checkpoints/lavida-o-v1.0
OUTOUT_PATH=outputs/geneval
VLM_PATH=none
DATASET=geneval 
export NCCL_TIMEOUT=7200
export NCCL_P2P_DISABLE=1

export N_GPUS=${N_GPUS:-"1"}

accelerate launch --main_process_port 30000 --num_processes $N_GPUS \
    -m eval_img.eval \
    --cfg 4.5 \
    --num_samples 1 \
    --schedule shift \
    --ckpt $CKPT \
    --base_model_path $BASE_MODEL_PATH \
    --name 1024-${EVAL_RUN}-${EVAL_CKPT} \
    --shift 7 \
    --steps 64 \
    --ema \
    --micro_cond "ORIGINAL WIDTH : 1024; ORIGINAL HEIGHT : 1024; TOP : 0; LEFT : 0; SCORE : 6.711" \
    --alg_temp 5 \
    --dynamic_temperature True \
    --min_temperature 0.2 \
    --top_p 1.0 \
    --top_k 0 \
    --cont \
    --res 1024 \
    --output_path $OUTOUT_PATH \
    --prefix "a realistic image, 4k. " \
    -i $ANNOTATION \
    --vlm_path $VLM_PATH \
    --dataset $DATASET \
    --fmt jpg \
    --config eval_img/eval_hd.yaml \
    ${@} \
