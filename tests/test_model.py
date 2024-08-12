import unittest
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch  # Import torch for creating attention masks


class TestModel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model = GPT2LMHeadModel.from_pretrained('../data/models/delmia_apriso_model')
        self.model.resize_token_embeddings(len(self.tokenizer))

    def test_generate(self):
        prompt = "What is Delmia Apriso?"
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # Create an attention mask
        outputs = self.model.generate(
            inputs,
            max_length=150,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Check for None tokens in the output
        tokens = outputs[0].tolist()
        if any(t is None for t in tokens):
            print("Warning: Found None token in output")

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
