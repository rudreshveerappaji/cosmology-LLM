import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Step 1: Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Prepare your dataset
# For simplicity, let's create a small in-memory dataset
cosmology_data = """
Question: What is the Big Bang theory?
Answer: The Big Bang theory is the prevailing cosmological model explaining the observable universe's origin from a singularity approximately 13.8 billion years ago.
Question: What is dark matter?
Answer: Dark matter is a type of matter hypothesized to account for approximately 85% of the matter in the universe that does not emit light or energy.
Question: What is a black hole?
Answer: A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it.
"""

# Save the dataset to a file
with open("cosmology_data.txt", "w") as f:
    f.write(cosmology_data)

# Step 3: Load the dataset
def load_dataset(file_path):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = load_dataset("cosmology_data.txt")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Step 4: Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Step 5: Save the fine-tuned model
model.save_pretrained("./cosmology_model")
tokenizer.save_pretrained("./cosmology_model")

# Step 6: Use the fine-tuned model to answer questions
def generate_answer(question):
    input_text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=150, num_return_sequences=1)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# Example usage
question = "What is the cosmic microwave background radiation?"
answer = generate_answer(question)
print(answer)