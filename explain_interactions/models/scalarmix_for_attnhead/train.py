import argparse
import numpy as np
import torch
from explain_interactions.utils import read_json_or_yml
from sklearn.metrics import precision_recall_fscore_support
from explain_interactions.registry import MODEL_REGISTRY
from explain_interactions.models import get_device
from datasets import load_dataset
from transformers import AutoTokenizer
from serpyco import Serializer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from explain_interactions.models.scalarmix_for_attnhead.expconfig import ExpConfig
from explain_interactions.models.scalarmix_for_attnhead.model import ClassifierWithAttnHeadDistribution
import os


def set_seed(_seed):
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(_seed)


def get_data(_config: ExpConfig, _tokenizer):
    if _config.hf_dataset == "snli":
        dataset = load_dataset("snli")
        dataset = dataset.filter(lambda example: example["label"] != -1)

        def tokenize_and_encode(batch):
            return _tokenizer(text=batch["premise"], text_pair=batch["hypothesis"])

        dataset_tokenized = dataset.map(tokenize_and_encode, batched=True)

    elif _config.hf_dataset == "fever":
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
        raise RuntimeError("only fever and snli is supported as dataset")
    return dataset_tokenized


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(y_pred=predictions, y_true=labels, average="macro")
    print(p, r, f1)
    return {"p": p, "r": r, "f1": f1}


def search_model(path):
    for root, dirs, _ in os.walk(path):
        for _dir in dirs:
            user_input = input(f"Found model dir {_dir}, do you want to use it? (y/n)")
            if user_input == "y":
                return os.path.join(root, _dir)
    raise RuntimeError("No model found")


def search_model_automatic(path):
    return os.path.join(path, sorted(os.listdir(path))[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="config_file", required=True)
    parser.add_argument("--mode", help="train/test", default="train", choices=["train", "test"])
    args = parser.parse_args()
    ser = Serializer(ExpConfig)
    config = ser.load(read_json_or_yml(args.config_file))
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_tokenizer)
    dataset_tokenized = get_data(config, tokenizer)
    print("data loaded, preparing model..")
    device = get_device()

    if args.mode == "train":
        expl_intr_model = MODEL_REGISTRY[config.model.init_params.expl_intr_model].load()
        model = ClassifierWithAttnHeadDistribution(
            orig_model=expl_intr_model, num_attn_heads=config.model.init_params.num_attn_heads
        )
        print("model loaded, training..")
    elif args.mode == "test":
        model = ClassifierWithAttnHeadDistribution.load(
            search_model_automatic(config.train_params.output_path),
            base_model_name=config.model.init_params.expl_intr_model,
            num_attn_heads=config.model.init_params.num_attn_heads,
        )
        print(type(model))
        model.to(device)
        print("model loaded, testing..")
    tok_max_length = tokenizer.model_max_length if config.hf_tokenizer != "prajjwal1/bert-small" else 128
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=tok_max_length, return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=config.train_params.output_path,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=config.train_params.batch_size,
        per_device_eval_batch_size=config.train_params.batch_size,
        learning_rate=config.train_params.lr,
        num_train_epochs=config.train_params.epochs,
        lr_scheduler_type=config.train_params.lr_scheduler_type,
        weight_decay=config.train_params.weight_decay,
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
    else:
        trainer.train()
        trainer.evaluate(dataset_tokenized["test"])
