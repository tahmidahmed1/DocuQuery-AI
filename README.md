# DocuQuery-AI

DocuQuery-AI is a Flask web app that lets users upload PDFs, extract text from each page using a multimodal LLM, store embeddings, and ask questions grounded in the uploaded documents.

## Features

- Upload one or multiple PDF files from a browser
- Extract structured text from PDF pages via image transcription
- Store processed contexts with vector embeddings in a local CSV database
- Retrieve the most relevant context with cosine similarity
- Generate contextual Q&A responses and display estimated API cost

## Tech Stack

- Python
- Flask
- OpenAI API (`gpt` + `text-embedding`)
- Pandas + NumPy
- PyPDF2
- pdf2image + Pillow

## Project Structure

```
DocuQuery-AI/
├── app.py
├── chatutils.py
├── requirements.txt
├── .env.example
└── README.md
```

## Local Setup

1. Clone the repository:

	```bash
	git clone https://github.com/tahmidahmed1/DocuQuery-AI.git
	cd DocuQuery-AI
	```

2. Create and activate a virtual environment:

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

4. Configure environment variables:

	```bash
	cp .env.example .env
	```

	Add your OpenAI key and Flask secret to `.env`:

	```env
	OPENAI_API_KEY=your_openai_api_key_here
	FLASK_SECRET_KEY=replace_with_random_secret
	```

5. Run the app:

	```bash
	python app.py
	```

6. Open in browser:

	- http://127.0.0.1:5000

## System Requirement

`pdf2image` requires Poppler. On macOS:

```bash
brew install poppler
```

## Resume-ready Description

Built a retrieval-augmented PDF question-answering web application using Flask and OpenAI APIs. Implemented document ingestion, embedding-based context retrieval with cosine similarity, and prompt-grounded response generation to answer user questions from uploaded files.