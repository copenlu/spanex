"""Script for training a Transformer model."""
import argparse
import random

# from transformers import get_cosine_schedule_with_warmup
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from transformers import AdamW, get_constant_schedule_with_warmup
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu", action="store_true", default=False)
    parser.add_argument("--seed", help="Random seed", type=int, default=73)

    parser.add_argument("--dataset", help="Path to the train datasets", type=str, choices=["snli", "fever"])
    parser.add_argument("--model_path", help="Path where the model will be serialized", default="nli_bert", type=str)
    parser.add_argument("--hf_model_name", help="Path where the model will be serialized", default="roberta-base", type=str)

    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--lr", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--epochs", help="Epochs number", type=int, default=4)
    parser.add_argument("--mode", help="Mode for the script", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--push_to_hub", help="Flag for pushing the model to the HF hub.", action="store_true", default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    print(args, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    if args.dataset == "snli":
        dataset = load_dataset("snli")
        dataset = dataset.filter(lambda example: example["label"] != -1)

        def tokenize_and_encode(batch):
            return tokenizer(text=batch["premise"], text_pair=batch["hypothesis"])

        dataset_tokenized = dataset.map(tokenize_and_encode, batched=True)

    elif args.dataset == "fever":
        dataset = load_dataset("copenlu/fever_gold_evidence")
        fever_ids = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}

        def make_evidence_text(example):
            example["evidence_text"] = " ".join(
                [" ".join(sent[0].split("_")) + " " + sent[-1] for sent in example["evidence"]]
            ).lstrip()
            example["label"] = fever_ids[example["label"]]
            return example

        def filter_length(example):
            """
            adding this because we don't want to bother with overflowing materials
            :param example:
            :return:
            """
            if len(example["evidence_text"].split()) + len(example["claim"].split()) < 400:
                return True
            return False

        dataset = dataset.map(make_evidence_text)

        def tokenize_and_encode(batch):
            return tokenizer(text=batch["claim"], text_pair=batch["evidence_text"], truncation="only_second")

        def tokenize_and_encode_rev(batch):
            """
            I think the previous method needs to be changed, the `premise` is evidence_text and the `hypothesis` is
            the claim.
            :param batch:
            :return:
            """
            return tokenizer(text=batch["evidence_text"], text_pair=batch["claim"])

        dataset_tokenized = dataset.map(tokenize_and_encode_rev, batched=True).filter(lambda x: len(x["input_ids"]) <= 512)

    else:
        raise RuntimeError("something went wrong")
    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    if args.mode == "train":
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_name, num_labels=3).to(device)
    elif args.mode == "test":
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=3).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(y_pred=predictions, y_true=labels, average="macro")
        print(p, r, f1)
        return {"p": p, "r": r, "f1": f1}

    if args.mode == "test":
        args.model_path = f"{args.hf_model_name}-{args.dataset}"
    training_args = TrainingArguments(
        output_dir=args.model_path,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
    )
    if args.mode == "test":
        trainer.evaluate(dataset_tokenized["test"])
        if args.push_to_hub:
            model_name = f"sagnikrayc/{args.hf_model_name}-{args.dataset}"
            print(model_name)
            model.push_to_hub(model_name, use_temp_dir=True, exist_ok=True)
            tokenizer.push_to_hub(model_name, use_temp_dir=True, exist_ok=True)
    else:
        trainer.train()

"""
To train:
python explain_interactions/models/train.py --dataset snli --model_path snli_roberta_base_1e5 --hf_model_name roberta-base --lr 1e-5 --batch_size 16 --gpu

To test and upload:
python explain_interactions/models/train.py --dataset snli --model_path snli_roberta_base_2e5/checkpoint-34336 --hf_model_name roberta-base --lr 2e-5 --batch_size 16 --gpu --mode test --push_to_hub
"""
