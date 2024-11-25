from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from tokenization import tokenized_datasets, tokenizer

# GPT-2 modelini yuklash
model = GPT2LMHeadModel.from_pretrained("gpt2")

# O'quv sozlamalarini aniqlash
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Baholash har epochdan keyin amalga oshiriladi
    save_strategy="epoch",        # Model har epochdan keyin saqlanadi
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    load_best_model_at_end=True,  # Eng yaxshi modelni oxirida yuklash
    fp16=False
)

# Trainer ob'ektini yaratish
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Modelni o'qitish
trainer.train()
# Model va tokenni saqlash
trainer.save_model("./MyGPT2Chatbot")
tokenizer.save_pretrained("./MyGPT2Chatbot")
