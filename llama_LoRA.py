import os

os.environ['TRANSFORMERS_CACHE'] = '/storage/jwang27/cache/'

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,7"


import torch
import transformers
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
from torch.utils.data import DataLoader


device = torch.device("cuda:0")
print("Using device:", device)

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             cache_dir = "/storage/jwang27/cache/",
                                             load_in_8bit=True,
                                            device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


data = load_dataset("samsum")
data_train, data_test, data_val = data["train"], data["test"], data["validation"]

lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

tokenizer.add_special_tokens({"pad_token": "<PAD>"})
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

output_dir = "/storage/jwang27/orientation/fine_tune"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
per_device_eval_batch_size = 1
eval_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 10
logging_steps = 10
learning_rate = 5e-4
max_grad_norm = 0.3
max_steps = 50
warmup_ratio = 0.03
evaluation_strategy="steps"
lr_scheduler_type = "constant"

training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            evaluation_strategy=evaluation_strategy,
            save_steps=save_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=lr_scheduler_type,
            ddp_find_unused_parameters=False,
            eval_accumulation_steps=eval_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )

def generate_prompt(dialogue, summary=None, eos_token="</s>"):
  instruction = "Summarize the following:\n"
  input = f"{dialogue}\n"
  summary = f"Summary: {summary + ' ' + eos_token if summary else ''} "
  prompt = (" ").join([instruction, input, summary])
  return prompt

def formatting_func(prompt):
  output = []

  for d, s in zip(prompt["dialogue"], prompt["summary"]):
    op = generate_prompt(d, s)
    output.append(op)

  return output


trainer = SFTTrainer(
    model=model,
    train_dataset=data_train,
    eval_dataset=data_val,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args
)

# We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model(f"{output_dir}/final")
