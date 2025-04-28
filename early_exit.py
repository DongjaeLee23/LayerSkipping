from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import torch
from copy import deepcopy
from time import time
from tqdm import tqdm
import gc
import csv

prompt = "typing import List\ndef bucket_sort(A: List):"

checkpoint = "facebook/layerskip-codellama-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.memory_allocated())

max_new_tokens = 512
do_sample = True
top_p = 0.9
temperature = 0.6

warmup = 2
repeat = 5  # maybe lower if you want faster testing!

early_exit_values = [2, 4, 6, 8, 10, 12]  # <-- dynamic early exit values to test

# Load base model and tokenizer
config = LlamaConfig.from_pretrained(checkpoint)
model = LlamaForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(checkpoint, use_fast=False)

model.to(device)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to(device)

generation_config = {
    "max_new_tokens": max_new_tokens,
    "do_sample": do_sample,
    "top_p": top_p,
    "temperature": temperature,
    "pad_token_id": tokenizer.eos_token_id,
}

# Warmup the model
print("Warming up...")
for _ in tqdm(range(warmup)):
    _ = model.generate(**inputs, **generation_config)

# Result storage
results = []

print(torch.cuda.memory_allocated())
with torch.no_grad():
    for early_exit in early_exit_values:
        print(f"\n======== Early Exit: {early_exit} ========")

        # Create draft assistant model
        weights_memo = {id(w): w for w in model.parameters()}
        assistant_model = deepcopy(model, memo=weights_memo)
        assistant_model.model.layers = assistant_model.model.layers[:early_exit]
        del assistant_model.model.layers[early_exit:]
        assistant_model.to(device)

        # -------- Autoregressive Decoding --------
        print("Autoregressive Decoding...")
        total_time_auto = 0
        total_tokens_auto = 0
        for _ in tqdm(range(repeat)):
            start = time()
            outputs = model.generate(**inputs, **generation_config)
            total_time_auto += time() - start
            total_tokens_auto += outputs.numel()

        avg_time_auto = total_time_auto / repeat
        tokens_per_sec_auto = total_tokens_auto / total_time_auto

        # -------- Self-Speculative Decoding --------
        print("Self-Speculative Decoding...")
        total_time_spec = 0
        total_tokens_spec = 0
        for _ in tqdm(range(repeat)):
            start = time()
            outputs = model.generate(**inputs, **generation_config, assistant_model=assistant_model)
            total_time_spec += time() - start
            total_tokens_spec += outputs.numel()

        avg_time_spec = total_time_spec / repeat
        tokens_per_sec_spec = total_tokens_spec / total_time_spec

        # Save results
        results.append({
            "early_exit": early_exit,
            "auto_total_time": total_time_auto,
            "auto_avg_time": avg_time_auto,
            "auto_tokens_per_sec": tokens_per_sec_auto,
            "spec_total_time": total_time_spec,
            "spec_avg_time": avg_time_spec,
            "spec_tokens_per_sec": tokens_per_sec_spec,
        })
        # del model
        # del assistant_model
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

        # # Reload the model for next early_exit iteration
        # model = LlamaForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.float16)
        # model.to(device)


# -------- Final Report --------
print("\n\n========== FINAL RESULTS ==========")
for res in results:
    print(f"\nEarly Exit: {res['early_exit']}")
    print(f"  [Autoregressive]")
    print(f"    Total Time: {res['auto_total_time']:.2f} s")
    print(f"    Average Generation Time: {res['auto_avg_time']:.2f} s")
    print(f"    Average Tokens/sec: {res['auto_tokens_per_sec']:.2f}")
    print(f"  [Self-Speculative]")
    print(f"    Total Time: {res['spec_total_time']:.2f} s")
    print(f"    Average Generation Time: {res['spec_avg_time']:.2f} s")
    print(f"    Average Tokens/sec: {res['spec_tokens_per_sec']:.2f}")


csv_file = "benchmark_results_13B.csv"
csv_columns = [
    "early_exit",
    "auto_total_time",
    "auto_avg_time",
    "auto_tokens_per_sec",
    "spec_total_time",
    "spec_avg_time",
    "spec_tokens_per_sec"
]

with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for res in results:
        writer.writerow(res)

print(f"\nResults saved to {csv_file}")

from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
import torch
from copy import deepcopy
from time import time
from tqdm import tqdm
import gc
import csv

prompt = "typing import List\ndef bucket_sort(A: List):"

checkpoint = "facebook/layerskip-llama2-13B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.memory_allocated())

max_new_tokens = 512
do_sample = True
top_p = 0.9
temperature = 0.6

warmup = 2
repeat = 5  # maybe lower if you want faster testing!

early_exit_values = [2, 4, 6, 8, 10, 12]  # <-- dynamic early exit values to test

# Load base model and tokenizer
config = LlamaConfig.from_pretrained(checkpoint)
model = LlamaForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.float16)
tokenizer = LlamaTokenizer.from_pretrained(checkpoint, use_fast=False)

model.to(device)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(prompt, return_tensors="pt").to(device)

generation_config = {
    "max_new_tokens": max_new_tokens,
    "do_sample": do_sample,
    "top_p": top_p,
    "temperature": temperature,
    "pad_token_id": tokenizer.eos_token_id,
}

# Warmup the model
print("Warming up...")
for _ in tqdm(range(warmup)):
    _ = model.generate(**inputs, **generation_config)

# Result storage
results = []

print(torch.cuda.memory_allocated())
with torch.no_grad():
    for early_exit in early_exit_values:
        print(f"\n======== Early Exit: {early_exit} ========")

        # Create draft assistant model
        weights_memo = {id(w): w for w in model.parameters()}
        assistant_model = deepcopy(model, memo=weights_memo)
        assistant_model.model.layers = assistant_model.model.layers[:early_exit]
        del assistant_model.model.layers[early_exit:]
        assistant_model.to(device)

        # -------- Autoregressive Decoding --------
        print("Autoregressive Decoding...")
        total_time_auto = 0
        total_tokens_auto = 0
        for _ in tqdm(range(repeat)):
            start = time()
            outputs = model.generate(**inputs, **generation_config)
            total_time_auto += time() - start
            total_tokens_auto += outputs.numel()

        avg_time_auto = total_time_auto / repeat
        tokens_per_sec_auto = total_tokens_auto / total_time_auto

        # -------- Self-Speculative Decoding --------
        print("Self-Speculative Decoding...")
        total_time_spec = 0
        total_tokens_spec = 0
        for _ in tqdm(range(repeat)):
            start = time()
            outputs = model.generate(**inputs, **generation_config, assistant_model=assistant_model)
            total_time_spec += time() - start
            total_tokens_spec += outputs.numel()

        avg_time_spec = total_time_spec / repeat
        tokens_per_sec_spec = total_tokens_spec / total_time_spec

        # Save results
        results.append({
            "early_exit": early_exit,
            "auto_total_time": total_time_auto,
            "auto_avg_time": avg_time_auto,
            "auto_tokens_per_sec": tokens_per_sec_auto,
            "spec_total_time": total_time_spec,
            "spec_avg_time": avg_time_spec,
            "spec_tokens_per_sec": tokens_per_sec_spec,
        })
        # del model
        # del assistant_model
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())

        # # Reload the model for next early_exit iteration
        # model = LlamaForCausalLM.from_pretrained(checkpoint, config=config, torch_dtype=torch.float16)
        # model.to(device)


# -------- Final Report --------
print("\n\n========== FINAL RESULTS ==========")
for res in results:
    print(f"\nEarly Exit: {res['early_exit']}")
    print(f"  [Autoregressive]")
    print(f"    Total Time: {res['auto_total_time']:.2f} s")
    print(f"    Average Generation Time: {res['auto_avg_time']:.2f} s")
    print(f"    Average Tokens/sec: {res['auto_tokens_per_sec']:.2f}")
    print(f"  [Self-Speculative]")
    print(f"    Total Time: {res['spec_total_time']:.2f} s")
    print(f"    Average Generation Time: {res['spec_avg_time']:.2f} s")
    print(f"    Average Tokens/sec: {res['spec_tokens_per_sec']:.2f}")


csv_file = "benchmark_results_13B.csv"
csv_columns = [
    "early_exit",
    "auto_total_time",
    "auto_avg_time",
    "auto_tokens_per_sec",
    "spec_total_time",
    "spec_avg_time",
    "spec_tokens_per_sec"
]

with open(csv_file, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()
    for res in results:
        writer.writerow(res)

print(f"\nResults saved to {csv_file}")