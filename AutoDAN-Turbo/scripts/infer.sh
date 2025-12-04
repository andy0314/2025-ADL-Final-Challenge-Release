CUDA_VISIBLE_DEVICES=0 \
python test.py \
    --strategy_library logs_r/Qwen3-Embedding-0.6B/lifelong_strategy_library.pk \
    --epochs 3 \
    --break_score 1.0 \
    --output_file ../results/autodan-turbo/public-public/gemma-1.1-2b-it/no_train-epoch3.jsonl
