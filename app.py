import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Microsoft DialoGPT-medium pretrained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function for responding with chat history
def respond(message, history=[]):
    # Tokenize user input
    new_input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors='pt')
    # Build input: append to previous chat history
    bot_input_ids = torch.cat([torch.tensor(history, dtype=torch.long), new_input_ids], dim=-1) if history else new_input_ids
    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    # Update history
    history = chat_history_ids[0].tolist()
    # Decode and return response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, history

# Create Gradio interface
iface = gr.ChatInterface(respond)
iface.launch()
