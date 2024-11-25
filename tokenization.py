from datasets import load_dataset
from transformers import AutoTokenizer

mydataset = load_dataset("daily_dialog")
print(mydataset)

# Datasetni yuklash
tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Datasetni tayyorlash: labelsni input_ids dan nusxalash
def preprocess_function(examples):
    inputs = tokenizer(examples["dialog"][0], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = inputs["input_ids"].copy()  # Labelsni input_ids bilan bir xil qilish
    return inputs

tokenized_datasets = mydataset.map(preprocess_function, batched=True, remove_columns=mydataset["train"].column_names)
