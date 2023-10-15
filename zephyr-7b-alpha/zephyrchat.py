import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

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
    # Initialize system message
    system_message = """You are a friendly chatbot who always responds in less than 500 words.
     You will be presented with input prompts from the user and you will respond to them. Here are examples of the messages you will see:

    System: <system message - important information that should not be ignored>
    User: <user message - the prompts from the user for you to answer>
    Bot: <bot response>

    You should respond to the user message with a fitting response. You can use the chat history to help you respond.

    You should only process the user message and generate the bot response. 

    You should never generate the system message or the user message. You should only generate the bot response.

    You should never generate the same response twice. If you generate the same response twice, you will be disqualified.

    You should never end your response in the middle of a sentence - all responses should end in a period, question mark, or exclamation point. If you end your response in the middle of a sentence, you will be disqualified.

    You should never generate a response that is not in response to the user message. If you generate a response that is not in response to the user message, you will be disqualified.

    You should never generate a response that contains User:, Bot:, or System:. If you generate a response that contains User:, Bot:, or System:, you will be disqualified.
    
    """
    
    # Initialize chat history with system message
    chat_history_ids = tokenizer.encode(f"System: {system_message}\n", return_tensors="pt").to(device)

    user_message = ""
    while user_message.lower() != "quit":
        user_message = input("User: ")
        
        new_input_ids = tokenizer.encode(f"User: {user_message}\nBot:", return_tensors="pt").to(device)
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

        # Prepare for incremental generation
        generated_response_ids = torch.tensor([], dtype=torch.long).to(device)
        next_input = chat_history_ids

        for i in range(500):  # Limit to 500 cycles to avoid infinite loop; adjust as needed
            with torch.no_grad():
                output = model.generate(next_input, max_length=next_input.shape[-1] + 10, pad_token_id=tokenizer.eos_token_id)

            # Extract newly generated tokens
            new_tokens = output[:, next_input.shape[-1]:]

            # Stop if the model produces an EOS token
            if tokenizer.eos_token_id in new_tokens:
                break

            # Print new tokens
            chat_output = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            print(chat_output + " ", end='', flush=True)

            # Update generated_response_ids and next_input
            generated_response_ids = torch.cat([generated_response_ids, new_tokens[0]], dim=0)
            next_input = torch.cat([chat_history_ids, generated_response_ids.unsqueeze(0)], dim=-1)
        
        print()  # Newline at end of response

        # Update chat history
        chat_history_ids = next_input

if __name__ == "__main__":
    chat_with_bot()
