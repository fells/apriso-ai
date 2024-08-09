from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def evaluate_model(model, tokenizer, prompt):
    # Tokenize the input prompt with padding
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)

    # Generate a response from the model with attention mask
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=50,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set
        )

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model_dir = '../data/models/delmia_apriso_model'

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)

    test_prompt = "This is a test prompt to evaluate the model."
    response = evaluate_model(model, tokenizer, test_prompt)

    print("Model response:", response)
