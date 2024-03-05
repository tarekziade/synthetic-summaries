from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd

model_name = "Qwen/Qwen1.5-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.to(device)

dataset = load_dataset("c4", "en", split="train[:0.001%]")


def generate_summary(text):
    prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.replace(prompt, "").strip()


# Create new dataset with summaries
new_data = []
for entry in dataset:
    print(f"Summarizing entry {dataset.index}: ")
    summary = generate_summary(entry["text"])
    new_entry = {
        "url": entry["url"],
        "timestamp": entry["timestamp"],
        "text": entry["text"],
        "summary": summary,
    }
    new_data.append(new_entry)

# Convert to DataFrame for easy handling and saving
new_dataset_df = pd.DataFrame(new_data)

# Save the new dataset to a CSV file
output_file = "c4-qwen1.5-7B-Chat-summary.csv"
new_dataset_df.to_csv(output_file, index=False)
print(f"Dataset saved successfully at {output_file}.")
