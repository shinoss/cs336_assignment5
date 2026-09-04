import torch
import sys
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PreTrainedTokenizer
from typing import Callable, Literal
import argparse
import json
from .vllm_utils import VLLMServer
from .checkpoint import get_model_and_tokenizer
from .drgrpo_grader import r1_zero_reward_fn
import logging

logger = logging.getLogger(__name__)

CLIP_VALUES = {
    "grpo": 0.2,
    "gspo": 3e-4,
}

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_and_output_strs = [p + " " + out for (p,out) in zip(prompt_strs, output_strs)]
    prompt_and_output_tensor = tokenizer(
        prompt_and_output_strs,
        padding=True,
        return_tensors="pt"
    )['input_ids']
    input_ids = prompt_and_output_tensor[:,:-1]
    labels = prompt_and_output_tensor[:,1:]
    encoded_prompts = tokenizer(prompt_strs)['input_ids']
    response_start_indices = torch.tensor([(len(p)-1) for p in encoded_prompts])
    max_len = input_ids.shape[-1]
    mask = torch.arange(max_len)[None,:] >= response_start_indices[:,None]
    is_pad = labels != tokenizer.pad_token_id
    response_mask = mask * is_pad
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_logprobs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    out = model(input_ids)
    logits = out.logits # [B, T, vocab_size]
    all_log_probs = torch.log_softmax(logits, -1) # [B, T, vocab_size]
    log_probs = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1) # [B, T]
    token_entropy = {}
    if return_token_entropy:
        token_entropy = {
            "token_entropy": -(all_log_probs * all_log_probs.exp()).sum(-1)
        }
    return {
        "log_probs": log_probs,
        **token_entropy
    }


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    num_rollouts = len(rollout_responses)
    raw_rewards = torch.zeros(num_rollouts)

    i = 0
    total_reward = 0
    total_format_reward = 0

    for rollout, truth in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(rollout, truth)
        raw_rewards[i] = r['reward']
        total_reward += r['reward']
        total_format_reward += r['format_reward']
        i += 1

    metadata = {
        "reward": total_reward / num_rollouts,
        "format_reward": total_format_reward / num_rollouts,
    }

    return (raw_rewards, metadata)

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_rewards_per_group = raw_rewards.reshape(-1, group_size)

    group_mean = None
    if baseline == "mean":
        group_mean = raw_rewards_per_group.mean(-1, keepdim=True)

    if advantage_normalizer == "none":
        # Dr. GRPO divides
        group_normalization = 1
    elif advantage_normalizer == "std":
        # vanilla GRPO
        assert baseline == "mean"
        group_normalization = (raw_rewards_per_group - group_mean).std(-1, keepdim=True) + advantage_eps
    else:
        assert advantage_normalizer == "mean"
        group_normalization = raw_rewards_per_group.mean(-1, keepdim=True) + advantage_eps

    normalized_rewards_per_group = raw_rewards_per_group
    if group_mean is not None:
        normalized_rewards_per_group -= group_mean
    normalized_rewards_per_group /= group_normalization
    advantages = normalized_rewards_per_group.flatten(start_dim=0)

    metadata = {
        "mean": advantages.mean().item(),
        "min": advantages.min().item(),
        "max": advantages.max().item(),
    }
    return (advantages, metadata)

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantage = raw_rewards_or_advantages 
    if advantage.ndim == 1:
        advantage = advantage.unsqueeze(-1)

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -(policy_log_probs * advantage)
    else:
        assert old_log_probs is not None
        log_prob_factor = policy_log_probs - old_log_probs
        reweight_factor = torch.exp(log_prob_factor)
        if importance_reweighting_method == "noclip":
            per_token_policy_gradient_loss = -(reweight_factor * advantage)
        else:
            if importance_reweighting_method == "gspo":
                seq_lens = response_mask.sum(dim=-1,keepdim=True)
                reweight_factor = (log_prob_factor * response_mask).sum(dim=-1,keepdim=True) # [BxG,1]
                reweight_factor /= seq_lens
                reweight_factor = torch.exp(reweight_factor).expand_as(log_prob_factor)

            per_token_policy_gradient_loss = -1 * torch.minimum(
                advantage * reweight_factor,
                advantage * torch.clamp(reweight_factor, 1-cliprange, 1+cliprange)
            )
    metadata = {}
    return (per_token_policy_gradient_loss, metadata)

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "constant":
        # NOTE: we don't do mean() for Dr. GRPO case here because normalization constant
        # is B * G * max_gen_length, so denominator already contains B * G 
        per_rollout_loss = masked_loss / normalization_constant
        return per_rollout_loss.sum()
    else:
        # NOTE: important! need to ensure that we only average over the number of UN-masked elements
        # instead of total sequence length (i.e. T = all tokens in sequence)
        response_lengths = mask.sum(-1)
        per_rollout_loss = masked_loss.sum(-1) / response_lengths
        return per_rollout_loss.mean()

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    train_device = next(model.parameters()).device
    microbatch_size = len(repeated_prompts) // gradient_accumulation_steps
    total_loss = 0.0

    raw_rewards, reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    logger.debug(f"Raw rewards: {raw_rewards}")

    advantage, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards, 
        group_size, 
        baseline, 
        advantage_eps, 
        advantage_normalizer
    )
    logger.debug(f"Advantage: {advantage}")

    for start in range(0, len(repeated_prompts), microbatch_size):
        end = start + microbatch_size
        inputs = repeated_prompts[start:end]
        outputs = rollout_responses[start:end]

        tokenized_result = tokenize_prompt_and_output(inputs, outputs, tokenizer)
        input_ids = tokenized_result['input_ids'].to(train_device)
        labels = tokenized_result['labels'].to(train_device)
        mask = tokenized_result['response_mask'].to(train_device)

        response_logprobs = get_response_logprobs(model, input_ids, labels)
        log_probs = response_logprobs['log_probs']

        old_log_probs_microbatch = None
        if old_log_probs is not None:
            old_log_probs_microbatch = old_log_probs[start:end,:log_probs.shape[-1]].to(log_probs.device)

        microbatch_advantage = advantage[start:end].unsqueeze(-1).to(train_device)
        per_token_policy_gradient_loss, _ = compute_policy_gradient_loss(microbatch_advantage, log_probs, importance_reweighting_method, old_log_probs_microbatch, cliprange, mask)
        logger.debug(f"Per-token policy grad loss: {per_token_policy_gradient_loss}")

        microbatch_loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss, mask, loss_normalization, normalization_constant)
        if loss_normalization == "sequence":
            microbatch_loss /= gradient_accumulation_steps
        logger.debug(f"Microbatch loss adjusted: {microbatch_loss}")

        total_loss += microbatch_loss
        microbatch_loss.backward()

    # optional grad clip
    if max_grad_norm is not None:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        l2norm = torch.tensor(0.0, device=grads[0].device)
        for g in grads:
            l2norm += (g ** 2).sum()
        l2norm = l2norm.sqrt()
        downscale = min(1, max_grad_norm / (l2norm + 1e-6))
        for g in grads:
            g *= downscale

    optimizer.step()
    optimizer.zero_grad()

    metadata = {
        "avg_reward": reward_metadata["reward"] ,
        "avg_format_reward": reward_metadata["format_reward"],
        "mean_adv": advantage_metadata["mean"],
        "min_adv": advantage_metadata["min"],
        "max_adv": advantage_metadata["max"],
    }
    return (total_loss, metadata)


def get_dataset(ds_path: str, n_examples: int):
    with open(ds_path) as f:
        data = [json.loads(line) for line in f if line.strip()]
        questions = [d['question'] for d in data][:n_examples]
        answers = [d['answer'] for d in data][:n_examples]
    return questions, answers


# full training loop
def train_grpo(
    model_id: str,
    train_path: str,
    valid_path: str,
    prompt_path: str,
    group_size: int, # number of rollouts
    seed: int, 
    advantage_eps: float = 1e-6,
    baseline: Literal["mean", "none"] = "mean",
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
):
    torch.manual_seed(seed)

    n_train_examples = 6400
    n_val_examples = 1024
    num_rollout_steps = 200
    learning_rate = 1e-5
    rollout_batch_size = train_batch_size = 256
    prompts_per_batch = train_batch_size // group_size
    max_grad_norm = 1.0
    gradient_accumulation_steps = 32
    sampling_temperature = 1.0
    sampling_max_tokens = 512

    # Place train model on GPU 0 
    train_device = "cuda:0"
    policy, tokenizer = get_model_and_tokenizer(model_id, train_device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0)

    # Place vLLM inference model on GPU 1
    server = VLLMServer(model_id=model_id)
    server.start()
    server.init_weight_sync(train_device)
    sampling_params = {
        "temperature": sampling_temperature,
        "max_tokens": sampling_max_tokens,
        "n": group_size, # number of generations per prompt
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    cliprange = CLIP_VALUES.get(importance_reweighting_method, None)

    train_questions, train_answers = get_dataset(train_path, n_train_examples)
    valid_questions, valid_answers = get_dataset(valid_path, n_val_examples)

    template_name = prompt_path.split('/')[-1]
    print(f"Using prompt format: {template_name}")
    f = open(prompt_path)
    template = f.read()
    template_fn = lambda question: template.replace("{question}", question)
    templated_train_inputs = [template_fn(question) for question in train_questions]

    for step in range(num_rollout_steps):
        start = step * prompts_per_batch
        end = start + prompts_per_batch
        if end >= n_train_examples:
            break

        prompt_batch = templated_train_inputs[start:end]
        answer_batch = train_answers[start:end]

        # generate rollouts
        server.sync_policy_weights(policy)
        responses = server.generate_completions( # should have 256 responses (32 prompts x 8 prompts)
            prompt_batch,
            sampling_params,
            batch_size=rollout_batch_size, # in default set up, train batch size == inference batch size
        )
        rollout_responses = [resp.text for resp in responses]

        repeated_prompts = [prompt for prompt in prompt_batch for _ in range(group_size)]
        repeated_truths = [truth for truth in answer_batch for _ in range(group_size)]
        assert len(repeated_prompts) == len(rollout_responses)

        policy.train()
        # grade and do gradient descent step <grad_accumulation_steps> times
        total_loss = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer, 
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_truths,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=advantage_normalizer,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=None,
            cliprange=cliprange,
            loss_normalization=loss_normalization,
            normalization_constant=normalization_constant,
        )

        if step > 0 and step % 10 == 0:
            server.sync_policy_weights(policy)
            # run validation and log scores

        if step > 0 and step % 40 == 0:
            # Log example responses
            pass


def build_run_commands(args):
    return [
        [sys.executable, "-u", "scripts/grpo.py"]
    ]

def make_parser():
    parser = argparse.ArgumentParser("simple training script")
    parser.add_argument("--model-id", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--valid-path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt-path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    return parser

# from .modal_utils import app, submit_commands
# @app.local_entrypoint
# def modal_main(*argv: str) -> None:
#     args = make_parser().parse_args(list(argv))
#     commands = build_run_commands(args)
#     submit_commands(commands)


# def get_device():
#     if torch.backends.mps.is_available():
#         return "mps"
#     if torch.cuda.is_available():
#         return "cuda"
#     return "cpu"

if __name__ == "__main__":
    args = make_parser().parse_args()
    train_grpo(
        args.model_id,
        args.train_path,
        args.valid_path,
        args.prompt_path,
        args.group_size,
        args.advantage_eps,
    )
