from datasets import Dataset
from huggingface_hub import HfApi, HfFolder, Repository
import pandas as pd

"""
from huggingface_hub import HfApi
import getpass

# Get your Hugging Face username and password
username = input("Enter your Hugging Face username: ")
password = getpass.getpass("Enter your Hugging Face password: ")

# Authenticate with the Hugging Face Hub
api = HfApi()
token = api.login(username=username, password=password)

print("Logged in successfully!")
"""

# Load your data

#folder = "/mnt/InternalCrucial4/data/en-de/europarl/"
#filename = "europarl-v10.de-en"
folder = "/mnt/InternalCrucial4/data/en-de/paracrawl/"
filename = "paracrawlv9.en-de"



src = open(folder + filename + ".de", encoding="utf-8").read().splitlines()
tgt = open(folder + filename + ".en", encoding="utf-8").read().splitlines()
sco = open(folder + filename + ".cometkiwi22.sco", encoding="utf-8").read().splitlines()

# Create a Hugging Face Dataset
dataset = Dataset.from_dict({
    "de": src,
    "en": tgt,
    "sco": [float(s) for s in sco],
})

num_shards = 32

# Save each shard to a Parquet file
for i in range(num_shards):
    shard = dataset.shard(num_shards=num_shards, index=i)
    shard.to_parquet(f"{folder}train-{i:04d}-of-{num_shards:04d}.parquet")

"""
repo_id = "eole-nlp/filename"  # Replace with your username/repo name
dataset.push_to_hub(repo_id)
"""

