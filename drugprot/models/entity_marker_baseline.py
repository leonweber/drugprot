import os.path
from collections import defaultdict

import hydra.utils
import numpy as np
import torch
import pytorch_lightning as pl
import bioc
import torchmetrics
import transformers
from torch import nn
from tqdm import tqdm
from transformers.modeling_outputs import SequenceClassifierOutput

pl.seed_everything(42)

LABEL_TO_ID = {
    # DrugProt
    "NONE": 0,
    "ACTIVATOR": 1,
    "AGONIST": 2,
    "AGONIST-ACTIVATOR": 3,
    "AGONIST-INHIBITOR": 4,
    "ANTAGONIST": 5,
    "DIRECT-REGULATOR": 6,
    "INDIRECT-DOWNREGULATOR": 7,
    "INDIRECT-UPREGULATOR": 8,
    "INHIBITOR": 9,
    "PART-OF": 10,
    "PRODUCT-OF": 11,
    "SUBSTRATE": 12,
    "SUBSTRATE_PRODUCT-OF": 13,
    # ChemProt extra (not in DrugProt)
    # "COFACTOR": 14,
    # "DOWNREGULATOR": 15,
    # "INDIRECT-REGULATOR": 16,
    # "MODULATOR": 17,
    # "MODULATOR-ACTIVATOR": 18,
    # "MODULATOR-INHIBITOR": 19,
    # "NOT": 20,
    # "REGULATOR": 21,
    # "UNDEFINED": 22,
    # "UPREGULATOR": 23,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def insert_consistently(offset, insertion, text, starts, ends):
    new_text = text[:offset] + insertion + text[offset:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[starts >= offset] += len(insertion)
    new_ends[ends >= offset] += len(insertion)

    return new_text, new_starts, new_ends


def insert_pair_markers(text, head, tail, sentence_offset):
    starts = np.array(
        [head.locations[0].offset, tail.locations[0].offset]) - sentence_offset
    ends = starts + [head.locations[0].length, tail.locations[0].length]
    text, starts, ends = insert_consistently(offset=starts[0],
                                             text=text,
                                             insertion="[HEAD-S]",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=ends[0],
                                             text=text,
                                             insertion="[HEAD-E]",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=starts[1],
                                             text=text,
                                             insertion="[TAIL-S]",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=ends[1],
                                             text=text,
                                             insertion="[TAIL-E]",
                                             starts=starts,
                                             ends=ends
                                             )
    marked_head = text[text.index("[HEAD-S]"):text.index("[HEAD-E]")].replace("[HEAD-S]", "").replace("[HEAD-E]", "").replace("[TAIL-S]", "").replace("[TAIL-E]", "")
    marked_tail = text[text.index("[TAIL-S]"):text.index("[TAIL-E]")].replace("[HEAD-S]", "").replace("[HEAD-E]", "").replace("[TAIL-S]", "").replace("[TAIL-E]", "")
    assert (marked_head == head.text or not head.text.isalnum()) # skip unicode errors
    assert (marked_tail == tail.text or not tail.text.isalnum()) # skip unicode errors

    return text


def sentence_to_examples(sentence, tokenizer):
    examples = []

    ann_id_to_ann = {}
    for ann in sentence.annotations:
        ann_id_to_ann[ann.id] = ann

    pair_to_relations = defaultdict(set)
    for rel in sentence.relations:
        head = rel.get_node("head").refid
        tail = rel.get_node("tail").refid
        assert ann_id_to_ann[head].infons["type"] == "CHEMICAL" and "GENE" in \
               ann_id_to_ann[tail].infons["type"]
        rel = rel.infons["type"]
        pair_to_relations[(head, tail)].add(rel)

    for head in sentence.annotations:
        for tail in sentence.annotations:
            if not (head.infons["type"] == "CHEMICAL" and "GENE" in tail.infons["type"]):
                continue

            text = insert_pair_markers(text=sentence.text,
                                       head=head,
                                       tail=tail,
                                       sentence_offset=sentence.offset)


            features = tokenizer.encode_plus(text, max_length=512, truncation=True)

            try:
                assert "HEAD-S" in tokenizer.decode(features["input_ids"])
                assert "HEAD-E" in tokenizer.decode(features["input_ids"])
                assert "TAIL-S" in tokenizer.decode(features["input_ids"])
                assert "TAIL-E" in tokenizer.decode(features["input_ids"])
            except AssertionError:
                continue # entity was truncated


            features["labels"] = np.zeros(len(LABEL_TO_ID))
            for label in pair_to_relations[(head.id, tail.id)]:
                if label in LABEL_TO_ID:
                    features["labels"][LABEL_TO_ID[label]] = 1

            examples.append(
                {"head": head.id, "tail": tail.id, "features": features}
            )

    return examples


class Dataset:
    def __init__(self, path, tokenizer, limit_examples):
        self.examples = []
        self.tokenizer = tokenizer
        with open(hydra.utils.to_absolute_path(path)) as f:
            collection = bioc.load(f)
            for doc in tqdm(collection.documents, desc="Loading data"):
                if limit_examples and len(self.examples) > limit_examples:
                    break
                for passage in doc.passages:
                    for sentence in passage.sentences:
                        self.examples += sentence_to_examples(sentence, tokenizer)

    def __getitem__(self, item):
        return self.examples[item]["features"]

    def __len__(self):
        return len(self.examples)


class EntityMarkerBaseline(pl.LightningModule):
    def __init__(self, transformer: str, lr: float):
        super().__init__()
        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)
        self.tokenizer.add_tokens(["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"],
                                  special_tokens=True)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.transformer.config.hidden_size * 2,
                                    len(LABEL_TO_ID))
        self.train_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID), threshold=0.5)
        self.dev_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID), threshold=0.5)
        self.lr = lr
        self.num_training_steps = None
        self.collate_fn = transformers.DataCollatorWithPadding(self.tokenizer)

    def forward(self, features):
        output = self.transformer(input_ids=features["input_ids"],
                                  token_type_ids=features["token_type_ids"],
                                  attention_mask=features["attention_mask"]
                                  )
        seq_emb = self.dropout(output.last_hidden_state)
        head_idx = torch.where(
            features["input_ids"] == self.tokenizer.convert_tokens_to_ids("[HEAD-S]"))
        tail_idx = torch.where(
            features["input_ids"] == self.tokenizer.convert_tokens_to_ids("[TAIL-S]"))
        head_reps = seq_emb[head_idx]
        tail_reps = seq_emb[tail_idx]
        pairs = torch.cat([head_reps, tail_reps], dim=1)
        logits = self.classifier(pairs)
        if "labels" in features:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, features["labels"])
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )


    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.lr_schedulers().step()
        self.log("train/loss", output.loss, prog_bar=False)
        self.train_f1(torch.sigmoid(output.logits), batch["labels"].long())
        self.log("train/f1", self.train_f1, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("val/loss", output.loss, prog_bar=True)
        self.dev_f1(torch.sigmoid(output.logits), batch["labels"].long())
        self.log("val/f1", self.dev_f1, prog_bar=True)

        return output.loss

    def configure_optimizers(self):
        assert self.num_training_steps > 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedule = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=0.1 * self.num_training_steps,
                                                                num_training_steps=self.num_training_steps,
                                                                )

        return [optimizer], [schedule]

    def predict(self, collection: bioc.BioCCollection) -> None:
        threshold = 0.5
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        f1 = torchmetrics.F1()
        it = tqdm(collection.documents, desc="Predicting")
        for doc in it:
            for passage in doc.passages:
                for sent in passage.sentences:
                    examples = sentence_to_examples(sent, self.tokenizer)
                    if examples:
                        batch = collator([i["features"] for i in examples])
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)
                        out = self.forward(batch)
                        all_probs = torch.sigmoid(out.logits)
                        f1.update(all_probs.cpu(), batch['labels'].long().cpu())
                        if "labels" in batch:
                            it.set_description(f"F1: {f1.compute():.2f}")

                        for example, probs in zip(examples, all_probs):
                            head = example["head"]
                            tail = example["tail"]
                            for label_idx, prob in enumerate(probs):
                                if prob > threshold:
                                    rel = bioc.BioCRelation()
                                    rel.infons["prob"] = prob.item()
                                    rel.infons["type"] = ID_TO_LABEL[label_idx]
                                    rel.add_node(bioc.BioCNode(refid=head,
                                                               role="head"))
                                    rel.add_node(bioc.BioCNode(refid=tail,
                                                               role="tail"))
                                    sent.add_relation(rel)

        return None


    def get_dataset(self, path, limit_examples=None):
        return Dataset(path, self.tokenizer, limit_examples=limit_examples)
