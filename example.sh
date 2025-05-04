python bench.py --model Qwen/Qwen3-0.6B \
    --batch_sizes 1,2,4,8 \
    --context_lengths 512,1024,2048 \
    --warmup_iters 5 \
    --test_iters 20 \
    --output_csv qwen_latency.csv \
    --liger_kernel \