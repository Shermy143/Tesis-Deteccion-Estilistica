import datasets
import os
from transformers import AutoTokenizer

def load_and_prepare_dataset(dataset_name="dmitva/human_ai_generated_text", tokenizer_name="StyleDistance/mstyledistance", max_length=512):
    """
    Loads a dataset containing human and AI text pairs and formats it for sequence classification.
    """
    token = "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"
    os.environ["HF_TOKEN"] = token
    datasets.config.HF_TOKEN = token
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=token)
    
    print(f"Loading dataset: {dataset_name}")
    try:
        raw_dataset = datasets.load_dataset(dataset_name, token=token)
    except Exception as e:
        print(f"Failed to load dataset. Make sure you have internet access and the dataset name is correct. Error: {e}")
        return None, None

    # The dataset "dmitva/human_ai_generated_text" has 'human_text' and 'ai_text' columns.
    # We need to flatten this into a single 'text' column and a 'label' column (0 for human, 1 for AI).
    
    def unroll_examples(examples):
        texts = []
        labels = []
        
        # Depending on the dataset structure, we handle it
        if "human_text" in examples and "ai_text" in examples:
            for human_txt, ai_txt in zip(examples["human_text"], examples["ai_text"]):
                # Ensure text is string and not None
                if human_txt:
                    texts.append(human_txt)
                    labels.append(0) # 0 for human
                if ai_txt:
                    texts.append(ai_txt)
                    labels.append(1) # 1 for AI
        elif "text" in examples and "label" in examples:
            return {"text": examples["text"], "label": examples["label"]}
        else:
            raise ValueError(f"Dataset format not recognized. Features found: {examples.keys()}")
            
        return {"text": texts, "label": labels}

    print("Formatting dataset...")
    # Map raw paired dataset into flattened classification format
    # We batched the mapping to handle unrolling correctly
    unrolled_dataset = raw_dataset.map(unroll_examples, batched=True, remove_columns=raw_dataset["train"].column_names)
    
    # Split into train and validation if no validation split exists
    if "validation" not in unrolled_dataset and "test" not in unrolled_dataset:
        print("Splitting dataset into train and validation (90/10)...")
        # For memory efficiency on large datasets, maybe just take a slice
        # Using a subset for demonstration/speed if it's very large, but let's use all mapping
        split_dataset = unrolled_dataset["train"].train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
    else:
        train_dataset = unrolled_dataset["train"]
        val_dataset = unrolled_dataset.get("validation", unrolled_dataset.get("test"))

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    print("Tokenizing dataset...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Set format to PyTorch tensors
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_train, tokenized_val, tokenizer

if __name__ == "__main__":
    # Test run
    train_ds, val_ds, _ = load_and_prepare_dataset()
    if train_ds:
        print(train_ds[0])
