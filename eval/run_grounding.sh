CKPT=checkpoints/lavida-o-v1.0

accelerate launch --num_processes=1 llava/eval/predict_grounding.py \
    --model $CKPT ${@}