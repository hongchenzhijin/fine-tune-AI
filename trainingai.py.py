import os
import torch
import wandb
import multiprocessing
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments

def fine_tune():
    # Initialize wandb
    wandb.init(
        project="deepseek-vl-2-unsloth",
        name="ft_run",
        config={
            "learning_rate": 5e-6,
            "batch_size": 2,
            "epochs": 2
        }
    )

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        load_in_4bit=True,
        max_seq_length=2048,
        dtype=torch.bfloat16,
    )

    # Define training prompt style
    train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>
    {}
    </think>
    {}"""
    EOS_TOKEN = tokenizer.eos_token  # Define EOS_TOKEN which the model when to stop generating text during training
    EOS_TOKEN
    # Define formatting prompt function
    def formatting_prompts_func(examples):  # Takes a batch of dataset examples as input
        inputs = examples["Question"]       # Extracts the medical question from the dataset
        cots = examples["Complex_CoT"]      # Extracts the chain-of-thought reasoning (logical step-by-step explanation)
        outputs = examples["Response"]      # Extracts the final model-generated response (answer)
        
        texts = []  # Initializes an empty list to store the formatted prompts
        
        # Iterate over the dataset, formatting each question, reasoning step, and response
        for input, cot, output in zip(inputs, cots, outputs):  
            text = train_prompt_style.format(input, cot, output) + EOS_TOKEN  # Insert values into prompt template & append EOS token
            texts.append(text)  # Add the formatted text to the list

        return {
            "text": texts,  # Return the newly formatted dataset with a "text" column containing structured prompts
        }
    # Load dataset
    dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train[:500]", trust_remote_code=True)
    dataset_finetune = dataset.map(formatting_prompts_func, batched = True)
    dataset_finetune["text"][0]


    # Prepare model for training
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Define training arguments
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20,
        learning_rate=5e-6,
        logging_steps=1,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",
    )

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_finetune,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=args,
        packing=False,
        max_seq_length=2048,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model("final_model")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    process = multiprocessing.Process(target=fine_tune)
    process.start()
    process.join()
