import argparse
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_and_prepare_dataset

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

def train(args):
    model_name = "StyleDistance/mstyledistance"
    
    print("Preparing dataset...")
    # Load dataset
    train_dataset, val_dataset, tokenizer = load_and_prepare_dataset(
        dataset_name=args.dataset,
        tokenizer_name=model_name,
        max_length=args.max_length
    )
    
    if args.dummy_run:
        print("DUMMY RUN: Truncating datasets to 20 examples for fast verification.")
        train_dataset = train_dataset.select(range(min(20, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(10, len(val_dataset))))

    print("Loading model...")
    # Load mStyleDistance as a classification model
    # Note: AnnaWegmann/mStyleDistance is XLM-RoBERTa based, so it loads xlm-roberta architecture
    # We specify num_labels=2 for Human (0) and AI (1)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        ignore_mismatched_sizes=True, # Ignore size mismatch for the classification head since original is embedding
        token="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"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        # Reduce logging overhead
        logging_dir="./logs",
        logging_steps=50,
        # Important for Windows sometimes
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving explicitly to {args.output_dir}-final...")
    trainer.save_model(f"{args.output_dir}-final")
    tokenizer.save_pretrained(f"{args.output_dir}-final")
    print("Training finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dmitva/human_ai_generated_text", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="./mstyle-detector", help="Output directory for saved model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for tokenizer")
    parser.add_argument("--dummy_run", action="store_true", help="Run a fast minimal train loop to verify syntax and pipeline")
    
    args = parser.parse_args()
    train(args)
