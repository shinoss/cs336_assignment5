import argparse
import importlib
import json
import logging
from typing import Any, Callable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase

from .drgrpo_grader import r1_zero_reward_fn
from .modal_utils import app, submit_commands
from .vllm_utils import VLLMCompletion, VLLMServer

logger = logging.getLogger(__name__)

CLIP_VALUES = {
    "grpo": 0.2,
    "gspo": 3e-4,
}

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def initialize_experiment_logging(
    project: str | None,
    run_name: str | None,
    config: dict[str, Any],
) -> tuple[Any | None, Any | None]:
    """Initialize W&B when available, while always retaining stdlib logging."""
    if project is None:
        return None, None

    try:
        wandb = importlib.import_module("wandb")
        run = wandb.init(project=project, name=run_name, config=config)
        return wandb, run
    except Exception as exc:
        logger.warning("W&B is unavailable; falling back to console logging: %s", exc)
        return None, None


def log_metrics(metrics: dict[str, float], step: int, wandb_run: Any | None) -> None:
    """Log scalar metrics to the console and, when configured, W&B."""
    logger.info("step=%d metrics=%s", step, json.dumps(metrics, sort_keys=True))
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)


def log_rollouts(
    split: str,
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    step: int,
    wandb: Any | None,
    wandb_run: Any | None,
    max_examples: int = 8,
) -> None:
    """Log a small qualitative sample of model rollouts."""
    rows: list[list[Any]] = []
    for prompt, response, ground_truth in list(
        zip(prompts, responses, ground_truths)
    )[:max_examples]:
        reward = reward_fn(response, ground_truth)
        row = [
            prompt,
            response,
            ground_truth,
            reward["reward"],
            reward["format_reward"],
        ]
        rows.append(row)
        logger.info(
            "%s rollout step=%d reward=%.4f format_reward=%.4f\nprompt=%s\nresponse=%s",
            split,
            step,
            reward["reward"],
            reward["format_reward"],
            prompt,
            response,
        )

    if wandb is not None and wandb_run is not None and rows:
        table = wandb.Table(
            columns=["prompt", "response", "ground_truth", "reward", "format_reward"],
            data=rows,
        )
        wandb_run.log({f"{split}/rollouts": table}, step=step)


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    prompt_and_output_strs = [
        p + " " + out for (p, out) in zip(prompt_strs, output_strs)
    ]
    prompt_and_output_tensor = tokenizer(
        prompt_and_output_strs,
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    input_ids = prompt_and_output_tensor[:, :-1]
    labels = prompt_and_output_tensor[:, 1:]
    encoded_prompts = tokenizer(prompt_strs)["input_ids"]
    response_start_indices = torch.tensor([len(p) - 1 for p in encoded_prompts])
    max_len = input_ids.shape[-1]
    mask = torch.arange(max_len)[None, :] >= response_start_indices[:, None]
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
    logits = out.logits  # [B, T, vocab_size]
    all_log_probs = torch.log_softmax(logits, -1)  # [B, T, vocab_size]
    log_probs = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    token_entropy = {}
    if return_token_entropy:
        token_entropy = {
            "token_entropy": -(all_log_probs * all_log_probs.exp()).sum(-1)
        }
    return {"log_probs": log_probs, **token_entropy}


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    num_rollouts = len(rollout_responses)
    raw_rewards = torch.zeros(num_rollouts)

    i = 0
    total_reward = 0
    total_answer_reward = 0
    total_format_reward = 0

    for rollout, truth in zip(rollout_responses, repeated_ground_truths):
        r = reward_fn(rollout, truth)
        raw_rewards[i] = r['reward']
        total_reward += r['reward']
        total_format_reward += r['format_reward']
        total_answer_reward += r['answer_reward']
        i += 1

    metadata = {
        "reward": total_reward / num_rollouts,
        "answer_reward": total_answer_reward / num_rollouts,
        "format_reward": total_format_reward / num_rollouts,
    }

    return (raw_rewards, metadata)


def evaluate_policy(
    server: VLLMServer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: dict[str, Any],
    request_batch_size: int,
) -> tuple[dict[str, float], list[VLLMCompletion]]:
    """Generate and score one validation response per prompt."""
    if len(prompts) != len(ground_truths):
        raise ValueError("Validation prompts and ground truths must have the same length")
    if not prompts:
        raise ValueError("Validation requires at least one prompt")

    validation_sampling_params = dict(sampling_params)
    validation_sampling_params["n"] = 1
    completions = server.generate_completions(
        prompts,
        validation_sampling_params,
        batch_size=request_batch_size,
    )
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} validation completions, got {len(completions)}"
        )

    responses = [completion.text for completion in completions]
    _, reward_metadata = compute_rollout_rewards(reward_fn, responses, ground_truths)
    average_response_length = sum(
        len(completion.token_ids) for completion in completions
    ) / len(completions)
    metrics = {
        "reward": reward_metadata["reward"],
        "answer_reward": reward_metadata["answer_reward"],
        "format_reward": reward_metadata["format_reward"],
        "average_response_length": average_response_length,
    }
    return metrics, completions


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
        group_normalization = (
            raw_rewards_per_group - group_mean
        ).std(-1, keepdim=True) + advantage_eps
    else:
        assert advantage_normalizer == "mean"
        group_normalization = (
            raw_rewards_per_group.mean(-1, keepdim=True) + advantage_eps
        )

    normalized_rewards_per_group = raw_rewards_per_group.clone()
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

    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        per_token_policy_gradient_loss = -(policy_log_probs * advantage)
    else:
        assert old_log_probs is not None, "Importance reweighting requires logprobs of original policy"
        log_prob_factor = policy_log_probs - old_log_probs
        reweight_factor = torch.exp(log_prob_factor)
        if importance_reweighting_method == "noclip":
            per_token_policy_gradient_loss = -(reweight_factor * advantage)
        else:
            assert cliprange is not None
            if importance_reweighting_method == "gspo":
                assert response_mask is not None
                seq_lens = response_mask.sum(dim=-1, keepdim=True)
                reweight_factor = (log_prob_factor * response_mask).sum(
                    dim=-1, keepdim=True
                )  # [BxG, 1]
                reweight_factor /= seq_lens
                reweight_factor = torch.exp(reweight_factor).expand_as(log_prob_factor)

            metadata["clip_mask"] = torch.logical_or(
                reweight_factor < 1 - cliprange,
                reweight_factor > 1 + cliprange,
            )
            per_token_policy_gradient_loss = -1 * torch.minimum(
                advantage * reweight_factor,
                advantage
                * torch.clamp(reweight_factor, 1 - cliprange, 1 + cliprange),
            )
    return (per_token_policy_gradient_loss, metadata)


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask
    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required for constant loss normalization"
            )
        # We don't take a mean here because the normalizer already contains the
        # batch size (B * G) and maximum generation length.
        per_rollout_loss = masked_loss / normalization_constant
        return per_rollout_loss.sum()

    # Average over unmasked response tokens, rather than padded sequence length.
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
    batch_size = len(repeated_prompts)
    if not (
        batch_size == len(rollout_responses) == len(repeated_ground_truths)
    ):
        raise ValueError("Prompts, responses, and ground truths must have equal lengths")
    if batch_size == 0:
        raise ValueError("A training batch cannot be empty")
    if group_size <= 0 or batch_size % group_size != 0:
        raise ValueError("The training batch must contain complete reward groups")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "The training batch size must be divisible by gradient_accumulation_steps"
        )
    if loss_normalization == "constant" and normalization_constant is None:
        raise ValueError(
            "normalization_constant is required for constant loss normalization"
        )
    if importance_reweighting_method != "none":
        if old_log_probs is None:
            raise ValueError(
                "old_log_probs is required when importance reweighting is enabled"
            )
        if old_log_probs.ndim != 2 or old_log_probs.shape[0] != batch_size:
            raise ValueError(
                "old_log_probs must have shape (training_batch_size, sequence_length)"
            )

    train_device = next(model.parameters()).device
    microbatch_size = batch_size // gradient_accumulation_steps
    total_loss = 0.0
    entropy_sum = 0.0
    response_token_count = 0
    clipped_token_count = 0

    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )
    logger.debug("Raw rewards: %s", raw_rewards)

    advantage, advantage_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )
    logger.debug("Advantage: %s", advantage)

    for start in range(0, len(repeated_prompts), microbatch_size):
        end = start + microbatch_size
        inputs = repeated_prompts[start:end]
        outputs = rollout_responses[start:end]

        tokenized_result = tokenize_prompt_and_output(inputs, outputs, tokenizer)
        input_ids = tokenized_result["input_ids"].to(train_device)
        labels = tokenized_result["labels"].to(train_device)
        mask = tokenized_result["response_mask"].to(train_device)

        response_logprobs = get_response_logprobs(
            model,
            input_ids,
            labels,
            return_token_entropy=True,
        )
        log_probs = response_logprobs["log_probs"]
        token_entropy = response_logprobs["token_entropy"]
        entropy_sum += (token_entropy * mask).sum().detach().float().item()
        response_token_count += int(mask.sum().item())

        old_log_probs_microbatch = None
        if old_log_probs is not None:
            sequence_length = log_probs.shape[-1]
            if old_log_probs.shape[-1] < sequence_length:
                raise ValueError(
                    "old_log_probs is shorter than the tokenized training microbatch"
                )

            # The rollout batch and each training microbatch can have different
            # padded widths. Select the matching edge for the padding direction.
            if tokenizer.padding_side == "left":
                old_log_probs_microbatch = old_log_probs[
                    start:end, -sequence_length:
                ]
            else:
                old_log_probs_microbatch = old_log_probs[
                    start:end, :sequence_length
                ]
            old_log_probs_microbatch = old_log_probs_microbatch.to(
                log_probs.device
            )

        microbatch_advantage = advantage[start:end].unsqueeze(-1).to(train_device)
        per_token_policy_gradient_loss, policy_metadata = compute_policy_gradient_loss(
            microbatch_advantage,
            log_probs,
            importance_reweighting_method,
            old_log_probs_microbatch,
            cliprange,
            mask,
        )
        if "clip_mask" in policy_metadata:
            clipped_token_count += int(
                (policy_metadata["clip_mask"] * mask).sum().item()
            )
        logger.debug(
            "Per-token policy grad loss: %s", per_token_policy_gradient_loss
        )

        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss,
            mask,
            loss_normalization,
            normalization_constant,
        )
        if loss_normalization == "sequence":
            microbatch_loss /= gradient_accumulation_steps
        logger.debug("Microbatch loss adjusted: %s", microbatch_loss)

        total_loss += microbatch_loss.detach()
        microbatch_loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.tensor(0.0, device=train_device)
    for grad in grads:
        grad_norm += (grad ** 2).sum()
    grad_norm = grad_norm.sqrt()

    # optional grad clip
    if max_grad_norm is not None:
        downscale = min(1, max_grad_norm / (grad_norm + 1e-6))
        for g in grads:
            g *= downscale

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        "avg_reward": reward_metadata["reward"],
        "avg_answer_reward": reward_metadata["answer_reward"],
        "avg_format_reward": reward_metadata["format_reward"],
        "mean_adv": advantage_metadata["mean"],
        "min_adv": advantage_metadata["min"],
        "max_adv": advantage_metadata["max"],
        "grad_norm": grad_norm.detach().float().item(),
        "token_entropy": entropy_sum / max(response_token_count, 1),
        "clip_fraction": clipped_token_count / max(response_token_count, 1),
        "avg_response_length": response_token_count / len(rollout_responses),
        "entropy_sum": entropy_sum,
        "response_token_count": float(response_token_count),
        "clipped_token_count": float(clipped_token_count),
    }
    return (total_loss, metadata)


def extract_identity(ans: str) -> str:
    return ans

def extract_gsm8k(full_answer: str) -> str:
    num = full_answer.split("####")[-1].strip()
    num = num.replace(",","")
    return num

ANSWER_EXTRACTORS: dict[str, Callable[[str], str]] = {
    "identity": extract_identity,
    "gsm8k": extract_gsm8k,
}

def get_dataset(ds_path: str, n_examples: int, answer_extractor: Callable[[str],str] = extract_identity):
    with open(ds_path) as f:
        data = [json.loads(line) for line in f if line.strip()]
        questions = [d['question'] for d in data][:n_examples]
        answers = [answer_extractor(d['answer']) for d in data][:n_examples]
    return questions, answers


def compute_original_logprobs(
    model: PreTrainedTokenizer,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    responses: list[str],
    device: str,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        tokenized_result = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids = tokenized_result["input_ids"].to(device)
        labels = tokenized_result["labels"].to(device)
        log_probs = get_response_logprobs(model, input_ids, labels)["log_probs"].detach()
    model.train()
    return log_probs

def train_grpo(
    model_id: str,
    train_path: str,
    valid_path: str,
    prompt_path: str,
    group_size: int,  # number of rollouts
    seed: int,
    advantage_eps: float = 1e-6,
    baseline: Literal["mean", "none"] = "mean",
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
    validation_interval: int = 10,
    rollout_log_interval: int = 40,
    wandb_project: str | None = "cs336-a5-grpo",
    wandb_run_name: str | None = None,
    num_rollout_steps: int = 50,
    rollout_batch_size: int = 256,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 32,
    max_grad_norm: float | None = 1.0,
    sampling_temperature: float = 1.0,
    sampling_max_tokens: int = 512,
    answer_extractor: Literal["identity", "gsm8k"] = "gsm8k",
    vllm_gpu_memory_utilization: float = 0.75,
) -> None:
    torch.manual_seed(seed)

    n_train_examples = 6400
    n_val_examples = 1024
    learning_rate = 1e-5
    if group_size <= 0 or rollout_batch_size <= 0 or train_batch_size <= 0:
        raise ValueError("Batch sizes and group_size must be positive")
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size")
    if rollout_batch_size % train_batch_size != 0:
        raise ValueError("rollout_batch_size must be divisible by train_batch_size")
    if train_batch_size % group_size != 0:
        raise ValueError("train_batch_size must contain complete reward groups")
    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
    if validation_interval <= 0 or rollout_log_interval <= 0:
        raise ValueError("Logging intervals must be positive")
    if num_rollout_steps <= 0:
        raise ValueError("num_rollout_steps must be positive")
    if max_grad_norm is not None and max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    if sampling_temperature < 0:
        raise ValueError("sampling_temperature must be non-negative")
    if sampling_max_tokens <= 0:
        raise ValueError("sampling_max_tokens must be positive")
    if answer_extractor not in ANSWER_EXTRACTORS:
        raise ValueError(
            f"Unknown answer extractor {answer_extractor!r}; "
            f"choose from {sorted(ANSWER_EXTRACTORS)}"
        )
    if not 0 < vllm_gpu_memory_utilization <= 1:
        raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")

    # this is the number of prompts per rollout batch (i.e inference)
    # e.g. if rollout bsz is 256 and group size is 8, then we have 32 prompts
    # each prompt generates 8 prompts for a total of 256 responses
    # NOTE: rollout bsz is correct here and not train bsz
    # during on-policy, rollout_bsz == train_bsz so no difference but
    # for off-policy, train_bsz < rollout_bsz, i.e. we do multiple train steps
    # (i.e. optimizer.step()) for a single rollout batch of rollout_bsz responses
    # so rollout_bsz responses get divided into batches of train_bsz and we do
    # a total of (rollout_bsz // train_bsz) number of grpo_train_step() calls
    prompts_per_batch = rollout_batch_size // group_size

    if loss_normalization == "constant" and normalization_constant is None:
        normalization_constant = train_batch_size * sampling_max_tokens

    run_config = {
        "model_id": model_id,
        "seed": seed,
        "n_train_examples": n_train_examples,
        "n_val_examples": n_val_examples,
        "num_rollout_steps": num_rollout_steps,
        "learning_rate": learning_rate,
        "rollout_batch_size": rollout_batch_size,
        "train_batch_size": train_batch_size,
        "group_size": group_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "sampling_temperature": sampling_temperature,
        "sampling_max_tokens": sampling_max_tokens,
        "max_grad_norm": max_grad_norm,
        "baseline": baseline,
        "advantage_normalizer": advantage_normalizer,
        "importance_reweighting_method": importance_reweighting_method,
        "loss_normalization": loss_normalization,
        "normalization_constant": normalization_constant,
        "answer_extractor": answer_extractor,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
    }
    wandb, wandb_run = initialize_experiment_logging(
        wandb_project,
        wandb_run_name,
        run_config,
    )

    # Place train model on GPU 0
    train_device = "cuda:0"
    policy, tokenizer = get_model_and_tokenizer(model_id, train_device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # Place vLLM inference model on GPU 1
    server = VLLMServer(
        model_id=model_id,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
    )
    sampling_params = {
        "temperature": sampling_temperature,
        "max_tokens": sampling_max_tokens,
        "n": group_size,  # number of generations per prompt
        "seed": seed,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }
    cliprange = CLIP_VALUES.get(importance_reweighting_method, None)

    try:
        server.start()
        server.init_weight_sync(train_device)

        extract_answer = ANSWER_EXTRACTORS[answer_extractor]
        train_questions, train_answers = get_dataset(
            train_path,
            n_train_examples,
            extract_answer,
        )
        valid_questions, valid_answers = get_dataset(
            valid_path,
            n_val_examples,
            extract_answer,
        )

        template_name = prompt_path.split("/")[-1]
        logger.info("Using prompt format: %s", template_name)
        with open(prompt_path) as prompt_file:
            template = prompt_file.read()
        templated_train_inputs = [
            template.replace("{question}", question) for question in train_questions
        ]
        templated_validation_inputs = [
            template.replace("{question}", question) for question in valid_questions
        ]

        def run_validation(log_step: int) -> None:
            server.sync_policy_weights(policy)
            validation_metrics, validation_completions = evaluate_policy(
                server,
                templated_validation_inputs,
                valid_answers,
                r1_zero_reward_fn,
                sampling_params,
                request_batch_size=rollout_batch_size,
            )
            log_metrics(
                {
                    "val/reward": validation_metrics["reward"],
                    "val/format_reward": validation_metrics["format_reward"],
                    "val/average_response_length": validation_metrics[
                        "average_response_length"
                    ],
                },
                log_step,
                wandb_run,
            )
            log_rollouts(
                "val",
                templated_validation_inputs,
                [completion.text for completion in validation_completions],
                valid_answers,
                r1_zero_reward_fn,
                log_step,
                wandb,
                wandb_run,
            )

        # Establish a baseline before the first optimizer update.
        run_validation(log_step=0)
        last_completed_step = 0
        last_validation_step = 0

        for step in range(num_rollout_steps):
            start = step * prompts_per_batch
            end = start + prompts_per_batch
            if end > len(templated_train_inputs):
                logger.info(f"Reached the end of train inputs ({len(templated_train_inputs)} items) at step: {step}")
                break

            prompt_batch = templated_train_inputs[start:end]
            answer_batch = train_answers[start:end]

            # Generate 8 responses for each of 32 prompts using the current policy.
            server.sync_policy_weights(policy)
            responses = server.generate_completions(
                prompt_batch,
                sampling_params,
                batch_size=prompts_per_batch, # technically not needed; len(prompt_batch) == prompts_per_batch
            )
            rollout_responses = [response.text for response in responses]
            repeated_prompts = [
                prompt for prompt in prompt_batch for _ in range(group_size)
            ]
            repeated_truths = [
                truth for truth in answer_batch for _ in range(group_size)
            ]
            if not (
                len(repeated_prompts)
                == len(rollout_responses)
                == len(repeated_truths)
                == rollout_batch_size
            ):
                raise RuntimeError(
                    "Rollout prompts, responses, and ground truths are misaligned"
                )

            old_log_probs = None
            if importance_reweighting_method != "none":
                # Compute fixed old-policy log probabilities for completions
                old_log_probs = compute_original_logprobs(
                    policy,
                    tokenizer,
                    repeated_prompts,
                    rollout_responses,
                    train_device,
                )
                logger.info(f"[Step {step}] old log probs shape: {old_log_probs.shape}")

            policy.train()
            train_losses: list[float] = []
            train_metadata_per_batch: list[dict[str, torch.Tensor | float]] = []
            for train_start in range(0, rollout_batch_size, train_batch_size):
                train_end = train_start + train_batch_size
                train_old_log_probs = None
                if old_log_probs is not None:
                    train_old_log_probs = old_log_probs[train_start:train_end]

                train_loss, train_metadata = grpo_train_step(
                    model=policy,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_grad_norm=max_grad_norm,
                    reward_fn=r1_zero_reward_fn,
                    repeated_prompts=repeated_prompts[train_start:train_end],
                    rollout_responses=rollout_responses[train_start:train_end],
                    repeated_ground_truths=repeated_truths[train_start:train_end],
                    group_size=group_size,
                    baseline=baseline,
                    advantage_eps=advantage_eps,
                    advantage_normalizer=advantage_normalizer,
                    importance_reweighting_method=importance_reweighting_method,
                    old_log_probs=train_old_log_probs,
                    cliprange=cliprange,
                    loss_normalization=loss_normalization,
                    normalization_constant=normalization_constant,
                )
                train_losses.append(train_loss.detach().float().item())
                train_metadata_per_batch.append(train_metadata)

            log_step = step + 1
            last_completed_step = log_step
            mean_metadata = {
                key: sum(float(metadata[key]) for metadata in train_metadata_per_batch)
                / len(train_metadata_per_batch)
                for key in (
                    "grad_norm",
                    "avg_reward",
                    "avg_format_reward",
                    "avg_response_length",
                )
            }
            response_token_count = sum(
                float(metadata["response_token_count"])
                for metadata in train_metadata_per_batch
            )
            token_entropy = sum(
                float(metadata["entropy_sum"])
                for metadata in train_metadata_per_batch
            ) / max(response_token_count, 1.0)
            clip_fraction = sum(
                float(metadata["clipped_token_count"])
                for metadata in train_metadata_per_batch
            ) / max(response_token_count, 1.0)
            train_metrics = {
                "train/loss": sum(train_losses) / len(train_losses),
                "train/gradient_norm": mean_metadata["grad_norm"],
                "train/token_entropy": token_entropy,
                "train/reward": mean_metadata["avg_reward"],
                "train/format_reward": mean_metadata["avg_format_reward"],
                "train/average_response_length": mean_metadata[
                    "avg_response_length"
                ],
                "train/clip_fraction": clip_fraction,
                "train/optimizer_steps_per_rollout": float(
                    rollout_batch_size // train_batch_size
                ),
            }
            log_metrics(train_metrics, log_step, wandb_run)

            if step == 0 or log_step % rollout_log_interval == 0:
                log_rollouts(
                    "train",
                    repeated_prompts,
                    rollout_responses,
                    repeated_truths,
                    r1_zero_reward_fn,
                    log_step,
                    wandb,
                    wandb_run,
                )

            if log_step % validation_interval == 0:
                run_validation(log_step)
                last_validation_step = log_step

        # Always report the final policy, even when the final step does not land
        # exactly on validation_interval.
        if last_completed_step != last_validation_step:
            run_validation(last_completed_step)
    finally:
        server.stop()
        if wandb_run is not None:
            wandb_run.finish()


def build_run_commands(args):
    command = [
        "python",
        "-u",
        "-m",
        "cs336_alignment.grpo",
        "--model-id",
        args.model_id,
        "--train-path",
        args.train_path,
        "--valid-path",
        args.valid_path,
        "--prompt-path",
        args.prompt_path,
        "--group-size",
        str(args.group_size),
        "--seed",
        str(args.seed),
        "--advantage-eps",
        str(args.advantage_eps),
        "--baseline",
        args.baseline,
        "--advantage-normalizer",
        args.advantage_normalizer,
        "--importance-reweighting-method",
        args.importance_reweighting_method,
        "--loss-normalization",
        args.loss_normalization,
        "--validation-interval",
        str(args.validation_interval),
        "--rollout-log-interval",
        str(args.rollout_log_interval),
        "--num-rollout-steps",
        str(args.num_rollout_steps),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--train-batch-size",
        str(args.train_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--sampling-temperature",
        str(args.sampling_temperature),
        "--sampling-max-tokens",
        str(args.sampling_max_tokens),
        "--answer-extractor",
        args.answer_extractor,
        "--vllm-gpu-memory-utilization",
        str(args.vllm_gpu_memory_utilization),
    ]
    if args.normalization_constant is not None:
        command.extend(
            ["--normalization-constant", str(args.normalization_constant)]
        )
    if args.wandb_project is not None:
        command.extend(["--wandb-project", args.wandb_project])
    if args.wandb_run_name is not None:
        command.extend(["--wandb-run-name", args.wandb_run_name])
    if args.disable_wandb:
        command.append("--disable-wandb")
    return [command]

def make_parser():
    parser = argparse.ArgumentParser("simple training script")
    parser.add_argument("--model-id", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train-path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--valid-path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt-path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--baseline", choices=["mean", "none"], default="mean")
    parser.add_argument(
        "--advantage-normalizer",
        choices=["std", "none", "mean"],
        default="std",
    )
    parser.add_argument(
        "--importance-reweighting-method",
        choices=["none", "noclip", "grpo", "gspo"],
        default="none",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=["sequence", "constant"],
        default="sequence",
    )
    parser.add_argument("--normalization-constant", type=int)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--rollout-log-interval", type=int, default=40)
    parser.add_argument("--num-rollout-steps", type=int, default=50)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)
    parser.add_argument(
        "--answer-extractor",
        choices=sorted(ANSWER_EXTRACTORS),
        default="gsm8k",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.75,
    )
    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser

@app.local_entrypoint(name="grpo")
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = make_parser().parse_args()
    train_grpo(
        model_id=args.model_id,
        train_path=args.train_path,
        valid_path=args.valid_path,
        prompt_path=args.prompt_path,
        group_size=args.group_size,
        seed=args.seed,
        advantage_eps=args.advantage_eps,
        baseline=args.baseline,
        advantage_normalizer=args.advantage_normalizer,
        importance_reweighting_method=args.importance_reweighting_method,
        loss_normalization=args.loss_normalization,
        normalization_constant=args.normalization_constant,
        validation_interval=args.validation_interval,
        rollout_log_interval=args.rollout_log_interval,
        wandb_project=None if args.disable_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        num_rollout_steps=args.num_rollout_steps,
        rollout_batch_size=args.rollout_batch_size,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        sampling_temperature=args.sampling_temperature,
        sampling_max_tokens=args.sampling_max_tokens,
        answer_extractor=args.answer_extractor,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
