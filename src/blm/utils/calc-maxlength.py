from transformers import AutoTokenizer
from datasets import load_from_disk
from prompter import Prompter
from log import log_message
import os
from tqdm import tqdm

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
prompter = Prompter(tok)
max_seen = 0

# data_path = "/rep/rabuhweidi/sft/SFTTraining-SemEval/data"
data_path = "/rep/rabuhweidi/sft/SFTTraining-SemEval/input_folder/eval"

try:
    dataset = load_from_disk(data_path)
except FileNotFoundError:
    log_message("ERROR", f"Test dataset not found at: {data_path}")
except Exception as e:
    log_message("ERROR", f"Error loading test dataset: {e}")

dataset = dataset.map(prompter, batched=True)
promptCount=0
# for example in tqdm(dataset, desc="Evaluating"):
#     prompt = example['prompt']
#     max_seen = max(max_seen, len(tok(prompt)['input_ids']))
# log_message('\nmax_seen:',max_seen)
# print('\nmax_seen:',max_seen)


long = sum(len(tok(p)["input_ids"]) > 512 for p in dataset["train"]["prompt"])
total = len(dataset["train"])
print(long, "/", total)