from transformers import pipeline

# Load pre-trained GPT-2 model and tokenizer
generator = pipeline('text-generation', model = 'gpt2')

# Generate text
prompt = "Singapore's fintech industry is"
result = generator(prompt, max_length = 50, num_return_sequences = 1)

# Print the result
print(result[0]['generated_text'])