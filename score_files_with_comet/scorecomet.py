import json
import comet


def read_parallel_files(file_path1, file_path2, batch_size=10000):
    data = []
    with open(file_path1, "r", encoding="utf-8") as file1, open(
        file_path2, "r", encoding="utf-8"
    ) as file2:
        batch_data = []
        for line1, line2 in zip(file1, file2):

            entry = {"src": line1, "mt": line2}
            batch_data.append(entry)
            if len(batch_data) >= batch_size:
                data.append(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            data.append(batch_data)
    return data


def score_batch(model, batch_data):
    model_output = model.predict(batch_data, batch_size=256, gpus=1)
    return model_output


if __name__ == "__main__":
    file_path1 = "/mnt/InternalCrucial4/data/en-de/cc-matrix/cc-matrix-ende.original.en"
    file_path2 = "/mnt/InternalCrucial4/data/en-de/cc-matrix/cc-matrix-ende.original.de"
    output_file = (
        "/mnt/InternalCrucial4/data/en-de/cc-matrix/cc-matrix-ende.original.cometkiwi22.txt"
    )
    batch_size = 100000

    data = read_parallel_files(file_path1, file_path2, batch_size)
    model = comet.models.load_from_checkpoint(
        "/mnt/InternalCrucial4/nlp/comet-models/cometkiwi22/checkpoints/model.ckpt"
    ).half()
    batch_scores = []
    with open(output_file, "w", encoding="utf-8") as output:
        for batch_data in data:
            batch_scores = score_batch(model, batch_data).scores
            for score in batch_scores:
                output.write(f"{score:.4f}\n")
