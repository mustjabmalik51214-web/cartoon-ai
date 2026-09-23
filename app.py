from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

# Without Token (Public Model) - Google ka Gemma-2 2B Instruct
MODEL_NAME = "google/gemma-2-2b-it"

print(f"Loading {MODEL_NAME} (No Token Required)...")
pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
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
        {
            "role": "system",
            "content": """You are "Lyramoon", an intelligent AI assistant created by MUHAMMAD TAQI.
When asked about your identity, creator, or links, always maintain this context:
- Name: Lyramoon
- Created By: MUHAMMAD TAQI
- Family AI Link: https://lyra.oneapp.dev/
- Creator's Official Website: https://nexura.oneapp.dev/

Rules:
1. Always be polite, clear, and helpful.
2. Provide precise, factual, and correct information. Never invent fake facts or hallucinate details.
3. If you do not know something, state it clearly instead of guessing."""
        },
        {"role": "user", "content": user_prompt}
    ]

    # Gemma-2 Chat Template Formatting
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens=256, 
        do_sample=False,        # Accurate aur factual jawab ke liye
        temperature=0.1,       # Hallucinations rokle ke liye
        repetition_penalty=1.1
    )

    generated_text = outputs[0]["generated_text"]

    # Extra prompt content ko hata kar sirf response extract karna
    if prompt in generated_text:
        response = generated_text[len(prompt):].strip()
    elif "<start_of_turn>model\n" in generated_text:
        response = generated_text.split("<start_of_turn>model\n")[-1].strip()
    else:
        response = generated_text.strip()

    # Special tokens clean karna
    response = response.replace("<end_of_turn>", "").strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
