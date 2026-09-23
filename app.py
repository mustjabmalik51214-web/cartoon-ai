from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Qwen ki jagah Google ka Gemma 1.1 2B Instruct model (No Token Required)
print("Loading Google Gemma 1.1 2B Model...")
pipe = pipeline(
    "text-generation",
    model="google/gemma-1.1-2b-it",
    torch_dtype=torch.float32,
    device_map="auto"
)
print("Model Loaded Successfully!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    user_prompt = data.get("prompt", "")

    if not user_prompt:
        return jsonify({"response": "Please enter a message."}), 400

    messages = [
        {"role": "system", "content": """You are "Lyramoon", an intelligent AI assistant created by MUHAMMAD TAQI.
When asked about your identity, creator, or links, always maintain this context:
- Name: Lyramoon
- Created By: MUHAMMAD TAQI
- Family AI Link: https://lyra.oneapp.dev/
- Creator's Official Website: https://nexura.oneapp.dev/

Rules:
1. Always be polite, clear, and helpful.
2. Provide precise, factual, and correct information. Never invent fake facts or hallucinate details.
3. If you do not know something, state it clearly instead of guessing."""},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Google Gemma format parsing update
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    elif "<start_of_turn>model\n" in generated_text:
        response = generated_text.split("<start_of_turn>model\n")[-1].strip()
    else:
        response = generated_text.strip()

    # Gemma special end token cleaning
    response = response.replace("<end_of_turn>", "").strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
