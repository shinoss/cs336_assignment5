from cs336_alignment.vllm_utils import VLLMServer, VLLMCompletion
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
import json 

prompts = [
    ("cs336_alignment/prompts/question_only.prompt", question_only_reward_fn),
    ("cs336_alignment/prompts/r1_zero.prompt", r1_zero_reward_fn),
    ("cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt", r1_zero_reward_fn),
]

BATCH_SIZE = 64

def extract_numeric(full_answer: str):
    num = full_answer.split("####")[-1].strip()
    num = num.replace(",","")
    return int(num)

def prepare_gsm8k():
    gsm8k_path = "data/gsm8k/test.jsonl"
    with open(gsm8k_path) as f:
        data = [json.loads(line) for line in f if line.strip()]
        questions = [d['question'] for d in data]
        answers = [extract_numeric(d['answer']) for d in data]
    return questions, answers

def eval():
    sampling_params = {
        "temperature": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": 42,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    questions, answers = prepare_gsm8k()

    server = VLLMServer("allenai/OLMo-2-0425-1B")
    server.start()

    for prompt_path, reward_fn in prompts:
        format_correct = 0
        answer_correct = 0
        correct = 0 
        incorrect = 0
        template_name = prompt_path.split('/')[-1]
        print(f"Prompt format: {template_name}")
        with open(prompt_path) as f:
            template = f.read()
            template_fn = lambda question: template.replace("{question}", question)
            templated_questions = [template_fn(question) for question in questions]
            responses: list[VLLMCompletion] = server.generate_completions(
                templated_questions,
                sampling_params,
                batch_size=BATCH_SIZE,
            )
            assert len(responses) == len(questions)
            print(f"Forwarded {len(questions)} questions, got {len(responses)} responses")
            rewards = [reward_fn(resp.text, ans) for resp, ans in zip(responses, answers)]

            # print some example (formatted) question, generated answer vs actual answer
            for i in range(3):
                print(f"Question: {templated_questions[i]}")
                print(f"Model Resposne: {responses[i].text}")
                print(f"Actual answer: {answers[i]}")

            for reward in rewards:
                format_reward = reward['format_reward']
                answer_reward = reward['answer_reward']
                reward = reward['reward']
                if format_reward == 1 and answer_reward == 1:
                    correct += 1
                elif answer_reward == 1:
                    answer_correct += 1
                elif format_reward == 1:
                    format_correct += 1
                else:
                    incorrect += 1
        print(f"{correct=}\t{answer_correct=}\t{format_correct=}\t{incorrect=}")
            


if __name__ == "__main__":
    eval()