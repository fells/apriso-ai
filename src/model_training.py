from transformers import GPT2Tokenizer, GPT2LMHeadModel, LineByLineTextDataset, DataCollatorForLanguageModeling, \
    Trainer, TrainingArguments
import os


def load_dataset(file_path, tokenizer):
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )


if __name__ == "__main__":
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Add a padding token to the tokenizer
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

    dataset = load_dataset(train_files[0], tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='../data/models',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model('../data/models/delmia_apriso_model')
