from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

def evaluate_model(model, tokenizer, prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Correct the path format for Windows
    model_dir = os.path.abspath('../data/models/delmia_apriso_model')

    # Load tokenizer and model from local directory
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True, from_safetensors=True)

    test_prompt = "This is a test prompt to evaluate the model."
    response = evaluate_model(model, tokenizer, test_prompt)

    print("Model response:", response)
