import os
import time

from embedding import get_document_vectors

# The target size of each text chunk in tokens, limited to your model input size
# For example, we use "gte-base" (https://huggingface.co/thenlper/gte-base) here
# to embedded our document and it's limit token is 512. We can set any number
# between 1 and 512. We use 512 here.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 512))

# Location of clean data (already run data.sh)
folder_path = "../../../data/"

# Use a list comprehension to get the list of "*.txt" files
file_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(folder_path)
    for file in files
    if file.endswith(".txt")
]

print(f"Found {len(file_paths)} files")

document_contents = []  # Store all the contents of the documents
document_paths = []  # Store all the paths of the documents

# Read all documents
for file_path in file_paths:
    with open(file_path, "r") as file:
        content = file.read()
        document_contents.append(content)
        document_paths.append(file_path)

# Start measuring time
start_time = time.time()

# Generate embedding vectors for document chunks
get_document_vectors(document_contents, document_paths, CHUNK_SIZE)

# Calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Time taken for tokenization and inference: {elapsed_time:.4f} seconds")
