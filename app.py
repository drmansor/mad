from flask import Flask, request, jsonify
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import openai

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("jordan_transactions.csv", sep="\t", header=None)
df.columns = ["ID", "Mall", "Branch", "Date", "Quantity", "Price", "Type", "Status"]
rows = df.apply(lambda row: " | ".join(row.astype(str)), axis=1).tolist()

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(rows, convert_to_tensor=False)

# Create FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# API route
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    q_embed = model.encode([question])[0]
    _, I = index.search([q_embed], 5)
    context = "\n".join([rows[i] for i in I[0]])

    prompt = f"""You are analyzing mall transactions. Use the following data to answer:
{context}

Question: {question}
Answer:"""

    # Use OpenAI API key from environment
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({"answer": response['choices'][0]['message']['content']})

# Render-compatible run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
