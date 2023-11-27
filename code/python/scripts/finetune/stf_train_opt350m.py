import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split
from pynvml import *
import pandas as pd
import random

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

# ### Bits and Bytes configuration for using quantization
compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
print("[] load model.")
model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-350m",
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True)
## TRAINING

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.5,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj","k_proj"] # obtained by the output of the model
)

model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir="/home/harpo/CEPH/LLM-models/checkpoints/",
    per_device_train_batch_size=48,
    gradient_accumulation_steps=12,
    num_train_epochs=3,
    optim='adamw_bnb_8bit',
    save_steps=200,
    fp16=True,
    logging_steps=100,
    learning_rate=2e-5,
    max_grad_norm=0.3,
    #max_steps=5000,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps"  
)

# ### Load Dataset
print("[] load dataset.")
dataset = pd.read_csv("/home/harpo/git-repos/llm-dga-detector/rawdata/domains_finetuning_families_v2.csv")
train_dataset, val_dataset = train_test_split(dataset, test_size=0.20, random_state=42)

train_dataset = Dataset.from_pandas(train_dataset)
val_dataset = Dataset.from_pandas(val_dataset)
# ### Format data for training

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['domain'])):
        text = f"#domain: {example['domain'][i]}\n#label: {example['family'][i]}"
        output_texts.append(text)
    return output_texts

instruction_template = "#domain:"
response_template = "#label:"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, 
                                           response_template=response_template, 
                                           tokenizer=tokenizer, 
                                           mlm=False
                                           )
print("[] Training.")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    #dataset_text_field="text",
    formatting_func= formatting_prompts_func,
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    data_collator= collator,
    
)
trainer.train()
model.save_pretrained("/home/harpo/CEPH/LLM-models/opt350-dga/lora-multiclass/")
## TODO code for merging
## Add some parameters
