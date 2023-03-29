
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM#, LLaMAForCausalLM, LLaMATokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


# Setting for A100 - For 3090
MICRO_BATCH_SIZE = 8  # change to 4 for 3090
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 2  # paper uses 3
LEARNING_RATE = 2e-5  # from the original paper
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

#pretrained_llm = "decapoda-research/llama-7b-hf"
pretrained_llm = "facebook/opt-6.7b"

model_dir = "models"
model_name = "alpaca7B-lora"

# load dataset and tokenize
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


#tokenizer = LLaMATokenizer.from_pretrained(pretrained_llm, add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained(pretrained_llm)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

print("Load dataset and tokenize...")

data = load_dataset("json", data_files="alpaca_data.json")
data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length",
    )
)
"""
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
"""

# fine tinue
print("Fine tuning...")
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
#model = LLaMAForCausalLM.from_pretrained(
model = AutoModelForCausalLM.from_pretrained(
    pretrained_llm,
    load_in_8bit=True,
    device_map="auto",
)
model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)


# statistics about ratio of trainable to total parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_steps=100,
        logging_steps=1,
        output_dir=model_dir,
        save_total_limit=1,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained(model_dir)
