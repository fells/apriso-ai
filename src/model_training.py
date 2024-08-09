from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import logging
from datasets import load_dataset  # Use the Datasets library instead of LineByLineTextDataset

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer if it doesn't have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Resize the model's token embeddings to include the new padding token
    model.resize_token_embeddings(len(tokenizer))

    processed_data_dir = '../data/processed/'
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"Directory {processed_data_dir} does not exist.")

    train_files = [os.path.join(processed_data_dir, file_name) for file_name in os.listdir(processed_data_dir) if
                   file_name.endswith('.txt')]
    if not train_files:
        raise ValueError("No training files found in the processed_data_dir. Ensure the directory contains .txt files.")

    # Load dataset using Datasets library
    dataset = load_dataset('text', data_files=train_files)
    dataset = dataset['train'].map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=128), batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='../data/models',
        overwrite_output_dir=True,
        num_train_epochs=5,  # Number of epochs
        per_device_train_batch_size=4,  # Batch size that fits within 12GB VRAM
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=500,
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        eval_steps=1000,
        weight_decay=0.01,
        learning_rate=5e-5,
        fp16=True,  # Mixed precision for faster training
        gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch size
        optim="adamw_torch",  # Use torch's AdamW for performance
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    # Save the model and tokenizer
    model_dir = '../data/models/delmia_apriso_model'
    trainer.save_model(model_dir)  # Saves the tokenizer too for compatible transformers versions
    tokenizer.save_pretrained(model_dir)
