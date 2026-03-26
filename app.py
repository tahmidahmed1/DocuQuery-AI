import os
from pathlib import Path

import numpy as np
import pandas as pd
import PyPDF2
from flask import Flask, flash, redirect, render_template_string, request, session, url_for
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from chatutils import generate_response, pdf_file_ingestion

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
PDFS_DB_PATH = BASE_DIR / "pdfs.csv"

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


def _parse_embedding(value: str) -> np.ndarray:
    cleaned = value.strip().strip("[]")
    if not cleaned:
        return np.array([], dtype=float)
    return np.fromstring(cleaned, sep=",")


def _load_pdf_dataframe() -> pd.DataFrame:
    if PDFS_DB_PATH.exists():
        data = pd.read_csv(PDFS_DB_PATH)
        if "embedding" in data.columns:
            data["embedding"] = data["embedding"].apply(_parse_embedding)
        return data
    return pd.DataFrame(columns=["filename", "context", "embedding"])


pdfs = _load_pdf_dataframe()
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

upload_template = """
<!doctype html>
<html>
    <head>
        <title>DocuQuery AI</title>
    </head>
    <body>
        <h1>Upload PDFs</h1>
        <form action="{{ url_for('upload') }}" method="post" enctype="multipart/form-data">
            <input type="file" name="pdf" multiple required>
            <input type="submit" value="Upload PDFs">
        </form>
        {% if 'pdf_contexts' in session %}
        <h2>Ask a Question</h2>
        <form action="{{ url_for('ask') }}" method="post">
            <input type="text" name="question" placeholder="Ask something..." required>
            <input type="submit" value="Ask">
        </form>
        {% if 'response' in session %}
        <h3>Response</h3>
        <p>{{ session['response'] }}</p>
        {% endif %}
        <form action="{{ url_for('reset') }}" method="post">
            <input type="submit" value="Upload New PDFs">
        </form>
        {% endif %}
        <br/>
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <ul>
                {% for message in messages %}
                    <li>{{ message }}</li>
                {% endfor %}
                </ul>
            {% endif %}
        {% endwith %}
    </body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(upload_template)


@app.route("/upload", methods=["POST"])
def upload():
    global pdfs

    if "pdf_contexts" not in session:
        session["pdf_contexts"] = []

    files = request.files.getlist("pdf")
    for file in files:
        if file.filename == "":
            flash("No selected file")
            continue

        if not (file and file.filename.lower().endswith(".pdf")):
            flash(f"Invalid file type. {file.filename} is not a PDF.")
            continue

        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)

        try:
            with file_path.open("rb") as file_obj:
                PyPDF2.PdfReader(file_obj)

            if filename in pdfs["filename"].values:
                flash(f"PDF is already in the database: {filename}.")
                if filename not in session["pdf_contexts"]:
                    session["pdf_contexts"].append(filename)
                continue

            context, embedding, cost = pdf_file_ingestion(str(file_path))
            flash(f"PDF {filename} uploaded and processed successfully. Cost estimate: ${cost:.2f}")

            text_file_path = file_path.with_suffix(".txt")
            text_file_path.write_text(context, encoding="utf-8")

            row = pd.DataFrame([[filename, context, embedding]], columns=pdfs.columns)
            pdfs = pd.concat([pdfs, row], ignore_index=True)
            pdfs.to_csv(PDFS_DB_PATH, index=False)

            session["pdf_contexts"].append(filename)
        except Exception:
            flash(f"Invalid PDF file: {filename}.")
            continue

    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    global pdfs

    if "pdf_contexts" not in session or not session["pdf_contexts"]:
        flash("No PDF contexts available. Please upload one or more PDFs first.")
        return redirect(url_for("index"))

    question = request.form["question"]
    filenames = session["pdf_contexts"]
    contexts = pdfs[pdfs["filename"].isin(filenames)][["context", "embedding"]].values.tolist()
    response, cost = generate_response(contexts, question)
    session["response"] = f"{response}\nEstimated Cost: ${cost:.2f}"
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("pdf_contexts", None)
    session.pop("response", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
