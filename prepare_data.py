# This code is adapted from `https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py`

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

""" Data Instances
{
   "text": "This is basically a peanut flavoured cream thickened with egg yolks and then set into a ramekin on top of some jam. Tony, one of the Wedgwood chefs, suggested sprinkling on some toasted crushed peanuts at the end to create extra crunch, which I thought was a great idea. The result is excellent.",
   "id": "<urn:uuid:e5a3e79a-13d4-4147-a26e-167536fcac5d>",
   "dump": "CC-MAIN-2021-43",
   "url": "<http://allrecipes.co.uk/recipe/24758/peanut-butter-and-jam-creme-brulee.aspx?o_is=SimilarRecipes&o_ln=SimRecipes_Photo_7>",
   "date": "2021-10-15T21:20:12Z",
   "file_path": "s3://commoncjcrawl/crawl-data/CC-MAIN-2021-43/segments/1634323583083.92/warc/CC-MAIN-20211015192439-20211015222439-00600.warc.gz",
   "language": "en",
   "language_score": 0.948729,
   "token_count": 69
}
"""

LOCAL_DATA_DIR = "edu_fineweb10B"
REMOTE_NAME = "sample-10BT"
SHARD_SIZE = int(1e8)  #  100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), LOCAL_DATA_DIR)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# Download the dataset
data = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train")

# Init tokenizer
encoder = tiktoken.get_encoding("gpt2")
eot = encoder._special_tokens["<|endoftext|>"]


def tokenize(doc):
    tokens = [eot]
    tokens.extend(encoder.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token out of range"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np, allow_pickle=True)


# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
npprocs = max(1, os.cpu_count() // 2)
with mp.Pool(npprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)

    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, data, chunksize=16):
        if token_count + len(tokens) < SHARD_SIZE:
            all_tokens_np[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(
                    total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_index}"
                )
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(
                DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}"
            )
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = SHARD_SIZE - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
