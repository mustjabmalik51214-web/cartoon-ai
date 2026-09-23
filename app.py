from flask import Flask, render_template, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)

print("Loading Qwen1.5 0.5B Chat Model...")
pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen1.5-0.5B-Chat",
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
    
    # Qwen1.5 Template Formatting
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens=256, 
        do_sample=False,        # Greedy decoding: Is se model sahi aur to-the-point jawab dega
        temperature=0.2,       # Low temperature keeps responses grounded
        repetition_penalty=1.1 # Repeating text ko rokne ke liye
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # Prompt ko hata kar sirf assistant ka response extract karna
    if prompt in generated_text:
        response = generated_text[len(prompt):].replace("<|im_end|>", "").strip()
    elif "<|im_start|>assistant\n" in generated_text:
        response = generated_text.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    else:
        response = generated_text.strip()

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
