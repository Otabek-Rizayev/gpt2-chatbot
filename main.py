from transformers import GPT2LMHeadModel, GPT2Tokenizer
from train_model import tokenizer

# Sozlangan modelni yuklash
model = GPT2LMHeadModel.from_pretrained("./MyGPT2Chatbot")
tokenizer = GPT2Tokenizer.from_pretrained("./MyGPT2Chatbot")

def chat_with_gpt2():
    print("Chatbot: Hello! Ask me anything or type 'exit' to leave.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Modeldan javob olish
        input_ids = tokenizer.encode(f"User: {user_input}\nBot:", return_tensors="pt")
        output = model.generate(input_ids, max_length=150, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, temperature=0.7)

        # Javobni chiqarish
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        bot_response = response.split("Bot:")[-1].strip()  # Faqat chatbotning javob qismini oling
        print(f"Chatbot: {bot_response}")

# Chatbotni ishga tushirish
chat_with_gpt2()
