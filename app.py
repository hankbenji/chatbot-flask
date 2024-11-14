from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained NLP model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "Please provide a message"}), 400
    
    # Tokenize user input and generate a response
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
    chat_history_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and send back the response
    bot_message = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": bot_message})

if __name__ == '__main__':
    app.run(debug=True)

