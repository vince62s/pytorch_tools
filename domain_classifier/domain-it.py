
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin


domains = ['Adult', 'Arts_and_Entertainment', 'Autos_and_Vehicles', 'Beauty_and_Fitness', 'Books_and_Literature', 'Business_and_Industrial', 'Computers_and_Electronics', 'Finance', 'Food_and_Drink', 'Games', 'Health', 'Hobbies_and_Leisure', 'Home_and_Garden', 'Internet_and_Telecom', 'Jobs_and_Education', 'Law_and_Government', 'News', 'Online_Communities', 'People_and_Society', 'Pets_and_Animals', 'Real_Estate', 'Science', 'Sensitive_Subjects', 'Shopping', 'Sports', 'Travel_and_Transportation']

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super(CustomModel, self).__init__()
        self.model = AutoModel.from_pretrained(config['base_model'])
        self.dropout = nn.Dropout(config['fc_dropout'])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config['id2label']))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)

# Setup configuration and model
config = AutoConfig.from_pretrained("nvidia/domain-classifier")
tokenizer = AutoTokenizer.from_pretrained("nvidia/domain-classifier")
model = CustomModel.from_pretrained("nvidia/domain-classifier").to("cuda")


def read_in_batches(src_path, tgt_path, sco_path, batch_size):
    with open(src_path, 'r') as srcfile, open(tgt_path, 'r') as tgtfile, open(sco_path, 'r') as scofile:
        srcbatch = []
        tgtbatch = []
        scobatch = []
        for srcline, tgtline, scoline in zip(srcfile, tgtfile, scofile):
            srcbatch.append(srcline.rstrip())  # Add line to batch, strip trailing newlines
            tgtbatch.append(tgtline.rstrip())  # Add line to batch, strip trailing newlines
            scobatch.append(scoline.rstrip())  # Add line to batch, strip trailing newlines
            if len(srcbatch) == batch_size:  # When the batch is full, yield it
                yield srcbatch, tgtbatch, scobatch
                srcbatch = []  # Reset batch
                tgtbatch = []
                scobatch = []
        if srcbatch:  # If there are leftover lines in the last batch
            yield srcbatch, tgtbatch, scobatch

# Example usage:
src_path = "/mnt/InternalCrucial4/data/en-de/paracrawl/paracrawlv9.en-de.100M.en"
tgt_path = "/mnt/InternalCrucial4/data/en-de/paracrawl/paracrawlv9.en-de.100M.de"
sco_path = "/mnt/InternalCrucial4/data/en-de/paracrawl/paracrawlv9.en-de.100M.cometkiwiXL.sco"
output_file = "/mnt/InternalCrucial4/data/en-de/paracrawl/paracrawlv9.en-de.100M."

src_path = "/mnt/InternalCrucial4/data/en-de/testsets/newstest2023-src.en"
tgt_path = "/mnt/InternalCrucial4/data/en-de/testsets/newstest2023-ref.de"
sco_path = "/mnt/InternalCrucial4/data/en-de/testsets/newstest2023.cometkiwi23.sco"
output_file = "/mnt/InternalCrucial4/data/en-de/testsets/newstest2023."

file_objects = {domain: open(output_file + domain, "w") for domain in domains}

batch_size = 8

for srcbatch, tgtbatch, scobatch in read_in_batches(src_path, tgt_path, sco_path, batch_size):
    inputs = tokenizer(srcbatch, return_tensors="pt", padding="longest", truncation=True).to("cuda")
    outputs = model(inputs['input_ids'], inputs['attention_mask'])

    # Predict and display results
    predicted_classes = torch.argmax(outputs, dim=1)
    predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes.cpu().numpy()]
    for i, domain in enumerate(predicted_domains):
        file_objects[domain].write(srcbatch[i] + "\t" + tgtbatch[i] + "\t" + scobatch[i] + "\n")
    
