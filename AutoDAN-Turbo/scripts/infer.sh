CUDA_VISIBLE_DEVICES=0 \
python test.py \
    --strategy_library logs/lifelong_strategy_library.pkl \
    --epochs 5 \
    --break_score 1.0 \
    --algorithm autodan-turbo/gemma-1.1-2b-it/no_train-epoch5 \
    --lm_combine_system_user
