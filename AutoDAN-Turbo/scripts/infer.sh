RUN_NAME=run1

CUDA_VISIBLE_DEVICES=0 \
python test.py \
    --strategy_library logs/${RUN_NAME}/lifelong_strategy_library.pkl \
    --epochs 5 \
    --break_score 1.0 \
    --algorithm autodan-turbo/${RUN_NAME}-epoch5 \
    --lm_combine_system_user
