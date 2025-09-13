CKPT=checkpoints/lavida-o-v1.0
CONFIG=llava/eval/1024_eval_edit.yaml

export NCCL_P2P_DISABLE=1
export N_GPUS=${N_GPUS:-"1"}
OUTOUT_PATH=outputs/edit-task-output
accelerate launch --main_process_port 30000 --num_processes $N_GPUS \
    -m eval_img.eval_imgedit \
    --config $CONFIG \
    --ckpt $CKPT \
    --output_path $OUTOUT_PATH \
    --name edit_result \
    ${@} \

