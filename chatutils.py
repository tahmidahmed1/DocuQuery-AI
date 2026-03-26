import base64
import os
from io import BytesIO
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from openai import OpenAI
from pdf2image import convert_from_path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def cosine_similarity(pdf_list: Sequence[Tuple[str, np.ndarray]], query: str) -> str:
    query_embedding = get_embedding(query)
    highest_similarity = 0.0
    most_similar_context = ""

    for context, embedding in pdf_list:
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_context = context

    return most_similar_context


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    if len(text) > 8191:
        text = text[:8191]
    cleaned = text.replace("\n", " ")
    return client.embeddings.create(input=cleaned, model=model).data[0].embedding


def pdf_to_base64(pdf_path: str) -> List[str]:
    images = convert_from_path(pdf_path, dpi=600)
    base64_images = []

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        base64_images.append(img_base64)

    return base64_images


def pdf_file_ingestion(pdf_path: str) -> Tuple[str, List[float], float]:
    base64_images = pdf_to_base64(pdf_path)
    structured_text = ""
    total_input_tokens = 0
    total_output_tokens = 0

    for base64_image in base64_images:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )

        structured_text += response.choices[0].message.content
        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

    input_cost = (total_input_tokens / 1_000_000) * 5.00
    output_cost = (total_output_tokens / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    return structured_text, get_embedding(structured_text), total_cost


def generate_response(contexts: Iterable[Tuple[str, np.ndarray]], question: str) -> Tuple[str, float]:
    context = cosine_similarity(contexts, question)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Forget all your knowledge and only use the following article to answer the question. If there is insufficient information in the article, say that you don't have enough information to answer the question.",
            },
            {"role": "user", "content": context},
            {"role": "user", "content": question},
        ],
    )

    cost = response.usage.total_tokens * 0.0001
    return response.choices[0].message.content, cost
