CUDA_VISIBLE_DEVICES=0 \
python main.py \
    --data data/public.json \
    --epochs 5 \
    --warm_up_iterations 1 \
    --lifelong_iterations 3 \
    --break_score 1.0 \
    --run_name run1 \
    --lm_combine_system_user
