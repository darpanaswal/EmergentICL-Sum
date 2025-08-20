from datasets import load_dataset

# Load WikiLingua dataset
dataset = load_dataset("GEM/wiki_lingua", "en")           # English articles -> English summaries
dataset_fr = load_dataset("GEM/wiki_lingua", "fr")        # French articles -> French summaries
dataset_cross = load_dataset("GEM/wiki_lingua", "fr_en")  # French articles -> English summaries

train = dataset["train"]
val = dataset["validation"]
test = dataset["test"]

train_fr = dataset_fr["train"]
val_fr = dataset_fr["validation"]
test_fr = dataset_fr["test"]

train_cross = dataset_cross["train"]
val_cross = dataset_cross["validation"]
test_cross = dataset_cross["test"]

train_cross = train_cross.filter(lambda example: example["source_language"] == "fr").filter(lambda example: example["target_language"] == "en")
val_cross = val_cross.filter(lambda example: example["source_language"] == "fr").filter(lambda example: example["target_language"] == "en")
test_cross = test_cross.filter(lambda example: example["source_language"] == "fr").filter(lambda example: example["target_language"] == "en")

# Sample only articles with <= 512 tokens
max_length = 512
def dataset_sample(dataset):
    return dataset.filter(lambda example: len(example["source"]) <= max_length)

train = dataset_sample(train)
val = dataset_sample(val)
test = dataset_sample(test)

train_fr = dataset_sample(train_fr)
val_fr = dataset_sample(val_fr)
test_fr = dataset_sample(test_fr)

train_cross = dataset_sample(train_cross)
val_cross = dataset_sample(val_cross)
test_cross = dataset_sample(test_cross)

# To csv files
train.to_csv("train.csv")
val.to_csv("val.csv")
test.to_csv("test.csv")

train_fr.to_csv("train_fr.csv")
val_fr.to_csv("val_fr.csv")
test_fr.to_csv("test_fr.csv")

train_cross.to_csv("train_cross.csv")
val_cross.to_csv("val_cross.csv")
test_cross.to_csv("test_cross.csv")
