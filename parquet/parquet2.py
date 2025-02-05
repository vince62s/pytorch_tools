from datasets import load_dataset, Dataset, Features, Value

# File paths
#folder = "/mnt/InternalCrucial4/data/en-de/paracrawl/"
#filename = "paracrawlv9.en-de"
#num_shards = 16  # Desired number of shards

#folder = "/mnt/InternalCrucial4/data/en-de/news-commentary/"
#filename = "news-commentary-v18.de-en"
#num_shards = 4  # Desired number of shards

#folder = "/mnt/InternalCrucial4/data/en-de/palm-synthetic/"
#filename = "greedy_decoded_en_de_sentence_level"
#num_shards = 4  # Desired number of shards

folder = "/mnt/InternalCrucial4/data/en-de/cc-matrix/"
filename = "cc-matrix-ende"
num_shards = 16  # Desired number of shards

# Function to count total rows in the source files
def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# Count the total rows in one of the files (all files should have the same number of lines)
total_rows = count_lines(folder + filename + ".de")
print(f"Total rows: {total_rows}")

# Calculate the number of rows per shard
shard_size = (total_rows + num_shards - 1) // num_shards  # Ceiling division
print(f"Shard size: {shard_size}")

# Load each file as streaming datasets
de_dataset = load_dataset("text", data_files={"de": folder + filename + ".de"}, split="de", streaming=True)
en_dataset = load_dataset("text", data_files={"en": folder + filename + ".en"}, split="en", streaming=True)
sco_dataset = load_dataset("text", data_files={"sco": folder + filename + ".cometkiwi22.sco"}, split="sco", streaming=True)

# Merge the datasets by zipping them together
def merge_rows(de_stream, en_stream, sco_stream):
#def merge_rows(de_stream, en_stream):
    for de_row, en_row, sco_row in zip(de_stream, en_stream, sco_stream):
    #for de_row, en_row in zip(de_stream, en_stream):
        yield {
            "de": de_row["text"].strip(),
            "en": en_row["text"].strip(),
            "sco": float(sco_row["text"].strip()),
        }

# Save shards incrementally
shard_data = {"de": [], "en": [], "sco": []}
current_shard = 0
current_row = 0

# Stream the merged dataset
for example in merge_rows(de_dataset, en_dataset, sco_dataset):
#for example in merge_rows(de_dataset, en_dataset):
    # Add the example to the current shard
    shard_data["de"].append(example["de"])
    shard_data["en"].append(example["en"])
    shard_data["sco"].append(example["sco"])
    current_row += 1

    # Save the shard when it reaches the shard_size or if it's the last shard
    if current_row == total_rows or len(shard_data["de"]) == shard_size:
        print(f"Saving shard {current_shard:04d} with {len(shard_data['de'])} rows.")
        shard = Dataset.from_dict(shard_data)
        shard.to_parquet(f"{folder}train-{current_shard:04d}-of-{num_shards:04d}.parquet")
        shard_data = {"de": [], "en": [], "sco": []}  # Reset for the next shard
        current_shard += 1

        # Stop saving if we've reached the desired number of shards
        if current_shard == num_shards:
            break

print(f"Dataset successfully saved in {current_shard} shards!")

