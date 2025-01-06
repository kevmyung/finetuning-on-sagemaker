import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from trl import SFTTrainer, setup_chat_format
from trl.commands.cli_utils import TrlParser
from peft import LoraConfig, PeftModel

# Constants
LLAMA_3_CHAT_TEMPLATE = """
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ message['content'] + eos_token }}
    {% elif message['role'] == 'assistant' %}
        {{ message['content'] + eos_token }}
    {% endif %}
{% endfor %}
"""

@dataclass
class ScriptArguments:
    train_dataset_path: str = field(default=None, metadata={"help": "Path to the train dataset"})
    validation_dataset_path: str = field(default=None, metadata={"help": "Path to the validation dataset"})
    model_id: str = field(default=None, metadata={"help": "Model ID to use for SFT training"})
    max_seq_length: int = field(default=512, metadata={"help": "The maximum sequence length for SFT Trainer"})

def load_and_process_datasets(script_args, tokenizer):
    def template_dataset(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False)}

    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.train_dataset_path, "train_dataset.json"),
        split="train"
    ).map(template_dataset, remove_columns=["messages"])

    validation_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.validation_dataset_path, "validation_dataset.json"),
        split="train"
    ).map(template_dataset, remove_columns=["messages"])

    return train_dataset, validation_dataset

def setup_model_and_tokenizer(script_args, training_args):
    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # Model setup
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        use_cache=not training_args.gradient_checkpointing
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer

def merge_and_save_model(model_id, adapter_dir, output_dir):
    """모델 병합 및 저장"""
    base_model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = peft_model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    base_model.config.save_pretrained(output_dir)

def training_function(script_args, training_args):
    """메인 학습 함수"""
    model, tokenizer = setup_model_and_tokenizer(script_args, training_args)
    
    train_dataset, validation_dataset = load_and_process_datasets(script_args, tokenizer)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
    )

    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()

    if training_args.distributed_state.is_main_process:
        merge_and_save_model(script_args.model_id, training_args.output_dir, "/opt/ml/model")
        tokenizer.save_pretrained("/opt/ml/model")

    training_args.distributed_state.wait_for_everyone()

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.local_rank == 0:
        print(f"SM_CURRENT_INSTANCE_TYPE: {os.getenv('SM_CURRENT_INSTANCE_TYPE')}")
        print(f"script_args: {script_args}")
        print(f"training_args: {training_args}")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    set_seed(training_args.seed)
    training_function(script_args, training_args)
