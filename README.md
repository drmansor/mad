# RAG CSV Transactions

This is a simple Retrieval-Augmented Generation (RAG) app that lets you query a CSV file of mall transactions using natural language.

## Features
- Loads a single CSV file
- Embeds each row as a document
- Answers questions using OpenAI's GPT-4 and top-k similar rows

## Setup

```bash
pip install -r requirements.txt
```

Add your OpenAI API key before running:
```python
import openai
openai.api_key = "YOUR_KEY"
```

## Run

```bash
python rag_csv_app.py
```
