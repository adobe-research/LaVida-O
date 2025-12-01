
# replace with s2 checkpoint
LLADA_8B_INSTRUCT=/mnt/localssd/lavida-llada-v1.0-instruct-gen-converted-v3-2048

bash /sensei-fs-3/users/shufanl/lavida-o-public/scripts/train/base.sh \
    --num_train_epochs 10  \
    --max_steps 100000  \
    --per_device_train_batch_size 2 \
    --data_path  config/data/debug_dataset.yaml \
    --model_name_or_path $LLADA_8B_INSTRUCT \
    --learning_rate 5e-6 \
    --dual_tower_layers 16 \
    --enc_use_image_branch  True \
    --group_by_random_length \
    --num_gen_image_tokens_enc_ds 1 \
    --image_enc_drop_rate 0.5 \
    --lmms_eval_generate_tasks="" \
    --mm_tunable_parts="mm_language_model_vision_parms" \
    --lmms_eval_extra_tasks "refcoco" 

sleep 2

