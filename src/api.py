from fastapi import FastAPI, Request
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI()

model_dir = './data/models/delmia_apriso_model'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_dir)

@app.post("/generate")
async def generate_prompt(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

# To run this API server, use the command:
# uvicorn api:app --reload
