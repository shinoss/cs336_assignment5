from transformers import AutoModelForCausalLM, AutoTokenizer  
import json 

def extract_numeric(full_answer: str):
    num = full_answer.split("####")[-1].strip()
    num = num.replace(",","")
    return int(num)

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

gsm8k_path = "data/gsm8k/test.jsonl"
with open(gsm8k_path) as f:
    data = [json.loads(line) for line in f if line.strip()]
    questions = [d['question'] for d in data][:1]
    answers = [extract_numeric(d['answer']) for d in data][:1]

inputs = tokenizer(questions, return_tensors='pt', return_token_type_ids=False, padding=True, padding_side='left')

pad_token = tokenizer.pad_token
olmo = olmo.to("mps")
inputs = {k: v.to("mps") for k,v in inputs.items()}

response = olmo.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=1.0, top_p=1.0)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])