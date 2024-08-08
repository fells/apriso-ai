from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

def evaluate_model(model, tokenizer, prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_dir = './data/models/delmia_apriso_model'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_dir)

    test_prompt = "What is Delmia Apriso used for?"
    response = evaluate_model(model, tokenizer, test_prompt)
    print(response)
