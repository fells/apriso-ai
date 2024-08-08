from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


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
    model_name = '../data/models/delmia_apriso_model'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    test_prompt = "This is a test prompt to evaluate the model."
    response = evaluate_model(model, tokenizer, test_prompt)

    print("Model response:", response)
