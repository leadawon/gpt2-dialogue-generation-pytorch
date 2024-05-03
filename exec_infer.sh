python src/main.py \
    --seed=0 \
    --mode="infer" \
    --data_dir="data" \
    --model_type="gpt2" \
    --bos_token="<bos>" \
    --sp1_token="<sp1>" \
    --sp2_token="<sp2>" \
    --gpu="0" \
    --max_len=1024 \
    --max_turns=5 \
    --top_p=0.8 \
    --ckpt_dir="saved_models" \
    --ckpt_name="best_ckpt_epoch=3_valid_loss=2.6631" \
    --end_command="Abort!"

python src/test.py