# Instructions

## Run on-policy GRPO
```
CS336_MODAL_GPU="L4:2" uv run modal run -m cs336_alignment.grpo \
  --num-rollout-steps 200 \
  --rollout-batch-size 32 \
  --train-batch-size 32 \
  --gradient-accumulation-steps 16 \
  --validation-interval 25 \
  --rollout-log-interval 40 \
  --sampling-max-tokens 512 \
  --vllm-gpu-memory-utilization 0.75 \
  --wandb-run-name grpo-l4-smoke-gradacc16
```

## Run off-policy GSPO