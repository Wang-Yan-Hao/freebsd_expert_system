# Retrieve plugin

This implementation utilizes an open-source embedded model ([gte-base](https://huggingface.co/thenlper/gte-base)) to convert the data into a set of vectors. Instead of storing the vectors in a vector database, we save them as a file due to the manageable size of the freebsd_data.

You can query the top TOP_K closest sentences to your question using cosine similarity. These retrieved sentences can provide additional context for your questions, thereby enhancing ChatGPT's ability to provide accurate answers.

Feel free to swap out the embedded model or the algorithm used for calculating similarities. Just keep in mind the token limit of the model. We choose the "gte-base" model due to its strong performance on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and its compact model size.

This project draws inspiration from the [chatgpt-retrieval-plugin](https://github.com/openai/chatgpt-retrieval-plugin) and achieves similar functionality. However, unlike the plugin, we do not rely on an online vector database. Nonetheless, it's worth mentioning that this project can be easily expanded to incorporate a vector database if necessary.

## Installation

``` shell
$ pip install -r requirements.txt
```

## Usage

### Embedded the freebsd data.
```shell
# Set hyperparameter
$ export CHUNK_SIZE=512 # The target size of each text chunk in tokens, limited to your model input size
$ export MIN_CHUNK_LENGTH_TO_EMBED=50 # Minimum length threshold for discarding  chunks, in tokens

# Run data.sh first to get clean data
$ sh data.sh

# Get embedding vectors
$ cd embedding
$ python main.py
```

The general step of `main.py` is to use a tokenizer to split the sentences into proper sizes (`CHUNK_SIZE`). Then, we use the model to embed the split sentences.

### Retrieve related question

```shell
# Set hyperparameter
$ export TOP_K=5 # Number of top-k vectors(sentences) to retrieve
$ export QUESTION="How to use the gunion command in FreeBSD?" # Question to retrieve

# Retrieve
$ cd query
$ python query.py
```

For those using the ChatGPT API, you can use `chatgpt.py` which requires an additional `OPENAIKEY` environment variable.

```shell
$ export OPENAIKEY="Your key"
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
