import os
import sys

sys.path.append("../")

import openai
from query import query

# Number of top-k vectors to retrieve
TOP_K = int(os.environ.get("TOP_K", 5))
QUESTION = os.environ.get("QUESTION", "How to use the gunion command in FreeBSD?")

openai.api_key = os.environ["OPENAIKEY"]


def ask_gpt(question: str, related_chunks: list):
    chunks_prompt_1 = "I will give you some related knowledge first."
    chunks_prompt_2 = "Yes."
    message = [
        {"role": "user", "content": chunks_prompt_1},
        {"role": "assistant", "content": chunks_prompt_2},
    ]

    for i in related_chunks:
        message.append({"role": "user", "content": i})
        message.append({"role": "assistant", "content": chunks_prompt_2})

    question_prompt = (
        "According to the above knowledge, answer the question: " + question
    )
    message.append({"role": "user", "content": question_prompt})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=message)

    print("--- GPT Answer: ---", response.choices[0].message["content"])


related_chunks = query(QUESTION)
ask_gpt(QUESTION, related_chunks)
