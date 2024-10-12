import numpy as np
from scipy.sparse import lil_matrix
from collections import defaultdict
from transformers import AutoTokenizer
from datasets import load_dataset
import fire
from tqdm import tqdm
import zlib

NUM_BUCKETS = 1000000


def hash_context(context):
    """
    Hash the context deterministically using zlib.crc32.

    Args:
        context: A tuple representing the n-1 context tokens (token IDs).

    Returns:
        A deterministic hash value modded by the number of buckets.
    """
    # Convert the context (tuple) to a string and encode to bytes
    context_bytes = str(context).encode("utf-8")

    # Compute the CRC32 hash and mod by NUM_BUCKETS
    return zlib.crc32(context_bytes) % NUM_BUCKETS


def build_ngram_model(corpus, n, tokenizer_name="bert-base-uncased"):
    if n not in [2, 3]:
        raise ValueError("n must be 2 (bigram), or 3 (trigram)")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the corpus and convert tokens to indexes
    token_ids = []
    for x, y in corpus:
        input_str = tokenizer.encode(x, add_special_tokens=False)[
            -n + 1 :
        ] + tokenizer.encode(y, add_special_tokens=False)
        token_ids.extend(input_str)

    vocab_size = tokenizer.vocab_size

    # Initialize the sparse matrix
    ngram_matrix = lil_matrix((NUM_BUCKETS, vocab_size))

    # Build n-gram counts
    ngram_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(token_ids) - n + 1):
        context = token_ids[i : i + n - 1]
        next_token = token_ids[i + n - 1]
        hashed_context = hash_context(tuple(context))
        ngram_counts[hashed_context][next_token] += 1

    # Fill the sparse matrix
    for hashed_context, next_token_counts in ngram_counts.items():
        for next_token, count in next_token_counts.items():
            ngram_matrix[hashed_context, next_token] = count

    # Normalize the matrix (convert counts to probabilities)
    row_sums = ngram_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    ngram_matrix = ngram_matrix.multiply(1 / row_sums.reshape(-1, 1))

    # Convert to COO format for efficient row slicing
    ngram_matrix = ngram_matrix.tocoo()

    return ngram_matrix, vocab_size, tokenizer


def predict_next_token(model, vocab_size, context):
    hashed_context = hash_context(tuple(context))

    # Extract the row corresponding to the hashed context
    row_mask = model.row == hashed_context
    col_indices = model.col[row_mask]
    data = model.data[row_mask]

    # Create a dense array of probabilities
    probabilities = np.zeros(vocab_size)
    probabilities[col_indices] = data

    if np.sum(probabilities) == 0:
        return None  # Indicate zero probability

    next_token_index = np.random.choice(vocab_size, p=probabilities)
    return next_token_index


def generate_tokens_autoregressively(model, vocab_size, context, k):
    generated_token_ids = []

    for _ in range(k):
        next_token_id = predict_next_token(model, vocab_size, context)
        if next_token_id is None:
            return None  # Indicate failure to generate sequence

        generated_token_ids.append(next_token_id)
        context = context[1:] + [next_token_id]

    return generated_token_ids


def main(train_file, out_file, n=2):
    dataset = load_dataset("json", data_files={"train": train_file})
    corpus = []

    for datum in dataset["train"]:
        corpus.append((datum["translation"]["en"], datum["translation"]["mentalese"]))

    ngram_matrix, vocab_size, tokenizer = build_ngram_model(
        corpus, n, tokenizer_name="openai-community/gpt2"
    )

    def make_ngram_metamer(example):
        encoded_en = tokenizer.encode(
            example["translation"]["en"], add_special_tokens=False
        )
        seed_len = len(
            tokenizer.encode(
                example["translation"]["mentalese"], add_special_tokens=False
            )
        )

        ngram_string = generate_tokens_autoregressively(
            ngram_matrix, vocab_size, encoded_en[-n + 1 :], seed_len
        )

        if ngram_string is not None:
            example["translation"]["mentalese"] = tokenizer.decode(ngram_string)

        return example

    updated_dataset = dataset.map(make_ngram_metamer)

    updated_dataset["train"].to_json(out_file)


if __name__ == "__main__":
    fire.Fire(main)
