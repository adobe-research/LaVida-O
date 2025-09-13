

CKPT=checkpoints/lavida-o-v1.0
LLADA_VISION_ENCODER="google/siglip-so400m-patch14-384"
set -x

export TASKS=${TASKS:-"mme,mmmu_val,mmbench_en_dev,textvqa_val,docvqa_val,chartqa,infovqa_val,scienceqa_full,ai2d,mathverse_testmini_vision_dominant,mathvista_testmini_format"}
export NOT_ALWASY_DO_2DPOOL=1
export DEBUG_PRINT_IMAGE_RES=1
export DEBUG_FIX_PADDING=1 # new runs must have this !!!!!!!!!!!!
echo $TASKS

export NCCL_P2P_DISABLE=1

accelerate launch --num_processes=1 \
    -m lmms_eval \
    --model llava_llada \
    --model_args pretrained=$CKPT,conv_template=llada,model_name=llava_llada \
    --tasks $TASKS \
    --batch_size 1 \
    --gen_kwargs prefix_lm=True \
    --log_samples \
    --log_samples_suffix llava_llada \
    --output_path outputs/und_logs/ --verbosity=DEBUG \
    --wandb_args=project=lmms-eval,job_type=eval,name=$EVAL_RUN-$EVAL_CKPT \
    ${@:2} \
