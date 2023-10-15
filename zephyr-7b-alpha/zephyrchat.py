import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Identify the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Initialize the model and tokenizer and move them to the device
print("Initializing model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16).to(device)

print("Model and tokenizer initialized.")

# Function to chat with the bot
def chat_with_bot():
    # Start with a system message
    system_message = "You are a friendly chatbot who always responds in the style of a pirate."
    user_message = ""
    
    # Tokenize the system message and move it to the device
    chat_history_ids = tokenizer.encode(f"System: {system_message}\n", return_tensors="pt").to(device)

    # Use a loop to keep the conversation going
    while user_message.lower() != "quit":
        # Get user input
        user_message = input("You: ")

        # Concatenate messages and tokenize
        new_message = f"You: {user_message}\nBot:"
        new_input_ids = tokenizer.encode(new_message, return_tensors="pt").to(device)  # Move new_input_ids to the device

        # Append new input to existing chat history
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1).to(device)  # Move new_input_ids to the device

        # Generate a response from the model
        with torch.no_grad():
            output = model.generate(chat_history_ids, max_length=256)

        # Extract the response and print it
        chat_output = tokenizer.decode(output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"Bot: {chat_output}")

        # Update chat history
        chat_history_ids = output

# Call function to start chatting
if __name__ == "__main__":
    chat_with_bot()
