import os
import re
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Minimum length threshold for discarding chunks, in tokens
# Hyperparameter that you can adjust
MIN_CHUNK_LENGTH_TO_EMBED = int(os.environ.get("MIN_CHUNK_LENGTH_TO_EMBED", 50))


# Utility Functions
def init_embedded_model():
    """
    Initialize the model, utilizing GPU if available.

    We use "gte-base" model here from huggingface, you
    can replace any model you want to embedded the document.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    global model
    model = SentenceTransformer(
        "thenlper/gte-base", device=device
    )  # https://huggingface.co/intfloat/e5-base-v2

    return model


def init_tokenizer():
    """
    Initialize the tokenizer.
    """
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "thenlper/gte-base"
    )  # https://huggingface.co/docs/transformers/internal/tokenization_utils


def get_vectors(text: str) -> np.array:
    """
    Generate vectors for the given text.
    """
    encoding = model.encode(
        [text], normalize_embeddings=True, batch_size=64
    )  # https://www.sbert.net/examples/applications/computing-embeddings/README.html
    return encoding


def store(path: str, data: List[str]):
    """
    Store data in the specified file with the given path.
    """
    with open(path, "w") as file:
        for string in data:
            file.write(string + "\n")


def process_text(text: str) -> str:
    """
    Process the text by removing extra whitespace.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Replace multiple newline characters with a single space
    text = re.sub(r"\n+", " ", text)
    return text


# Embedding Functions
def create_document_chunks(
    doc_content: str, chunk_token_size: Optional[int]
) -> List[str]:
    """
    Split the document content into chunks for embedding.

    This is just for split the sentences to suitable length.
    We still use the origin text to embedding.
    """
    if not doc_content or doc_content.isspace():
        return []

    # Process the space
    doc_content = process_text(doc_content)

    # Process with toeknizer
    tokens = tokenizer(
        doc_content,
        return_overflowing_tokens=True,
        truncation=False,
        padding=False,
        return_tensors="pt",
    )
    tokens = tokens.input_ids[0]
    tokens_list = []  # List to store the tokens

    # Split the tokens into the size of `chunk_token_size`
    while len(tokens) > MIN_CHUNK_LENGTH_TO_EMBED:
        token = tokens[:chunk_token_size]
        tokens_list.append(token)
        tokens = tokens[chunk_token_size:]

    return tokens_list


def get_document_vectors(
    document_contents: List[str], document_paths: List[str], chunk_token_size: int
):
    """
    Generate embedding vectors for document chunks and store them.
    """
    init_embedded_model()  # Init the model to embedding the documents
    init_tokenizer()

    # Store the chunk content
    chunks_contents: List[str] = []
    # Sote the path where the chunk come from, one-to-one with `chunk_contens` variable
    chunk_paths: List[str] = []

    # Loop over each document and create chunks
    for content, path in zip(document_contents, document_paths):
        document_chunks = create_document_chunks(content, chunk_token_size)
        # Store the return chunk
        chunks_contents.extend(document_chunks)
        chunk_paths.extend([path] * len(document_chunks))

    print(f"Total chunks from all documents: {len(chunks_contents)}")
    store("./chunk_paths.txt", chunk_paths)

    if not chunks_contents:
        return

    # Store the vectors
    embeddings_vectors: List[np.array] = []
    # Store the vectors origin texts, one-to-one with `embeddings_vectors` variable
    embeddings_texts: List[str] = []

    for i, chunk in enumerate(chunks_contents):
        if i % 10000 == 0:
            print(f"Processing chunk {i}")
        original_text = tokenizer.decode(
            chunk
        )  # Decode the chunk to the origin text for embedding
        vectors = get_vectors(original_text)  # Get the vectors
        embeddings_texts.append(original_text)
        embeddings_vectors.append(vectors)

    # Store the embedding as a file
    np.save("./embedding.npy", embeddings_vectors)
    store("./embedding_texts.txt", embeddings_texts)
