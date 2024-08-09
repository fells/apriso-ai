import unittest
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class TestModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('../data/models/delmia_apriso_model')

    def test_generate(self):
        prompt = "What is Delmia Apriso?"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=150, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main()
