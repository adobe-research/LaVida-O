
# replace with s2 checkpoint
LLADA_8B_INSTRUCT=/mnt/localssd/lavida-llada-v1.0-instruct-gen-converted-v3-2048

bash /sensei-fs-3/users/shufanl/lavida-o-public/scripts/train/s2-1024.sh \
    --num_train_epochs 10  \
    --max_steps 100000  \
    --per_device_train_batch_size 2 \
    --data_path  config/data/s1_und_gnd_data.yaml \
    --model_name_or_path $LLADA_8B_INSTRUCT \
    --learning_rate 2e-5 \
    --dual_tower_layers 16 \
    --enc_use_image_branch  True \
    --group_by_random_length \
    --num_gen_image_tokens_enc_ds 1 \
    --image_enc_drop_rate 0.5 \
    --lmms_eval_generate_tasks="" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_vision_resampler,mm_language_model" \
    --lmms_eval_extra_tasks "refcoco" 

sleep 2

