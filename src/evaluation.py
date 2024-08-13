from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def evaluate_model(model, tokenizer, prompt, max_length=50):
    # Tokenize the input prompt with padding and attention mask
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,  # Ensure pad_token_id is set
            eos_token_id=tokenizer.eos_token_id,  # Stop generation at the end of sequence
            do_sample=True,  # Enable sampling for more diverse outputs
            top_k=50,  # Use top-k sampling
            top_p=0.95,  # Use top-p (nucleus) sampling
        )

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model_dir = '../data/models/delmia_apriso_model'

    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True)

    # Test the model with a prompt
    test_prompt = "This is a test prompt to evaluate the model."
    response = evaluate_model(model, tokenizer, test_prompt)

    print("Model response:", response)
