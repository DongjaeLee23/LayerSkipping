from datasets import load_dataset
from transformers import AutoTokenizer
from gated_llama import GatedLlamaForCausalLM  # <- import your model
from torch.utils.data import DataLoader
from transformers import AdamW
import torch

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")  # small for testing
tokenizer = AutoTokenizer.from_pretrained("facebook/layerskip-llama2-7B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["article"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)
dataloader = DataLoader(dataset, batch_size=4)

# Load model
from transformers import AutoConfig
config = AutoConfig.from_pretrained("facebook/layerskip-llama2-7B")
model = GatedLlamaForCausalLM.from_pretrained("facebook/layerskip-llama2-7B", config=config)
model.train().cuda()

# Fine-tune
optimizer = AdamW(model.parameters(), lr=5e-5)
for epoch in range(3):
    for batch in dataloader:
        input_ids = batch["input_ids"].cuda()
        labels = batch["input_ids"].cuda()  # summarization = generation

        output = model(input_ids=input_ids, labels=labels)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")

model.save_pretrained("gated-layerskip-llama2")
tokenizer.save_pretrained("gated-layerskip-llama2")
