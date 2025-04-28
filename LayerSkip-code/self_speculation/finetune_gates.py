from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    default_data_collator,
    AdamW
)
from gated_llama import GatedLlamaForCausalLM
from torch.utils.data import DataLoader
import torch

# 1) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Load & preprocess dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")

tokenizer = AutoTokenizer.from_pretrained("facebook/layerskip-llama2-7B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["article"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# batched tokenization, drop original text column
dataset = dataset.map(tokenize, batched=True, remove_columns=["article", "highlights"])

# have HF return torch.Tensors directly for these columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 3) DataLoader with built‑in collator → tensors on batch
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=default_data_collator
)

# 4) Load model
config = AutoConfig.from_pretrained("facebook/layerskip-llama2-7B")
model = GatedLlamaForCausalLM.from_pretrained(
    "facebook/layerskip-llama2-7B",
    config=config,
    torch_dtype=torch.float16,   # <- very important
    device_map="auto"
)
model = model.to(device).train()
for param in model.model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    param.requires_grad = False
for param in model.gates.parameters():
    param.requires_grad = True

# 5) Optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
#optimizer = AdamW(model.parameters(), lr=5e-5)

# 6) Training loop
for epoch in range(3):
    print("Epoch: " + str(epoch))
    for i, batch in enumerate(dataloader):
        # now these are torch.Tensors, not lists
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # teacher‑forcing: use input_ids as labels
        labels         = batch["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i == 0 or i == len(dataloader) - 1:
            print(f"Loss: {loss.item():.4f}")

# 7) Save
model.save_pretrained("gated-layerskip-llama2")
tokenizer.save_pretrained("gated-layerskip-llama2")
