from flask import Flask, render_template, request, jsonify
import os
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

def get_groq_client():
    api_key = os.getenv("API_Key_GROQ")
    if not api_key:
        return None
    return Groq(api_key=api_key)

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    client = get_groq_client()
    if client is None:
        return jsonify({"error": "Server missing API_Key_GROQ environment variable"}), 500

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Act like a helpful personal Assistant"},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=512,
    )
    
    answer = response.choices[0].message.content.strip()
    return jsonify({"response": answer}), 200

@app.route("/summarize", methods=["POST"])
def summarize():
    email_text = request.form.get("email")
    if not email_text:
        return jsonify({"error": "Missing 'email' field"}), 400

    client = get_groq_client()
    if client is None:
        return jsonify({"error": "Server missing API_Key_GROQ environment variable"}), 500

    prompt = f"summarize the email in 2-3 sentences: {email_text}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Act like an expert email assistant"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=512,
    )

    summary = response.choices[0].message.content.strip()
    return jsonify({"response": summary}), 200

if __name__ == "__main__":
    app.run(debug=True)