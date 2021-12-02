import os.path
import warnings
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
import torch.nn.functional as F

from drugprot import utils
from drugprot.models.entity_marker_baseline import insert_pair_markers

log = utils.get_logger(__name__)


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
                assert "<e1>" in tokenizer.decode(features["input_ids"])
                assert "</e1>" in tokenizer.decode(features["input_ids"])
                assert "<e2>" in tokenizer.decode(features["input_ids"])
                assert "</e2>" in tokenizer.decode(features["input_ids"])
            except AssertionError:
                continue # entity was truncated


            features["labels"] = np.zeros(len(LABEL_TO_ID))
            for label in pair_to_relations[(head.id, tail.id)]:
                if label in LABEL_TO_ID:
                    features["labels"][LABEL_TO_ID[label]] = 1

            examples.append(
                {"head": head.id, "tail": tail.id, "features": features,
                 "head_cuid": head.infons["identifier"],
                 "tail_cuid": tail.infons["identifier"]
                 }
            )

    return examples


class Dataset:
    def __init__(self, path, tokenizer, limit_examples):
        self.doc_pair_to_examples = defaultdict(list)
        self.tokenizer = tokenizer
        with open(hydra.utils.to_absolute_path(path)) as f:
            collection = bioc.load(f)
            doc: bioc.BioCDocument
            for doc in tqdm(collection.documents, desc="Loading data"):
                if limit_examples and len(self.doc_pair_to_examples) > limit_examples:
                    break
                for passage in doc.passages:
                    for sentence in passage.sentences:
                        for example in sentence_to_examples(sentence, tokenizer):
                            doc_pair = (doc.id, example["head_cuid"], example["tail_cuid"])
                            self.doc_pair_to_examples[doc_pair].append(example)

        self.doc_pairs = sorted(self.doc_pair_to_examples.keys())

    def __getitem__(self, item):
        doc_pair = self.doc_pairs[item]
        return [i["features"] for i in self.doc_pair_to_examples[doc_pair]]

    def __len__(self):
        return len(self.doc_pairs)



class JointModel(pl.LightningModule):
    def __init__(self, transformer: str, lr: float, dist_penalize_factor: float):
        super().__init__()

        self.loss = nn.BCEWithLogitsLoss()
        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)
        self.tokenizer.add_tokens(['<e1>', '</e1>', '<e2>', '</e2>'],
                                  special_tokens=True)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.cuid_pair_classifier = nn.Linear(self.transformer.config.hidden_size * 2,
                                    len(LABEL_TO_ID))
        self.mention_pair_classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size * 4, self.transformer.config.hidden_size),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.transformer.config.hidden_size, len(LABEL_TO_ID))
        )
        self.train_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        self.dev_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        self.lr = lr
        self.num_training_steps = None
        self.thresholds = nn.Parameter(torch.ones(len(LABEL_TO_ID)) * 0.5)
        self.dist_penalize_factor = dist_penalize_factor

    def forward(self, features):
        output = self.transformer(input_ids=features["input_ids"],
                                  token_type_ids=features["token_type_ids"],
                                  attention_mask=features["attention_mask"]
                                  )
        seq_emb = self.dropout(output.last_hidden_state)
        head_idx = torch.where(
            features["input_ids"] == self.tokenizer.convert_tokens_to_ids('<e1>'))
        tail_idx = torch.where(
            features["input_ids"] == self.tokenizer.convert_tokens_to_ids('<e2>'))
        head_reps = seq_emb[head_idx]
        tail_reps = seq_emb[tail_idx]
        mention_pair_reps = torch.cat([head_reps, tail_reps], dim=1)

        cuid_pair_labels = torch.zeros((features["pair_ids"].max()+1, len(LABEL_TO_ID)),
                                       device=self.device)
        for i, mention_labels in enumerate(features["labels"]):
            cuid_pair_labels[features["pair_ids"][i]] += mention_labels
        cuid_pair_labels = torch.clamp(cuid_pair_labels, 0, 1)

        cuid_pair_reps = []
        for i, _ in enumerate(cuid_pair_labels):
            cuid_pair_reps.append(mention_pair_reps[features["pair_ids"] == i].mean(dim=0))
        cuid_pair_reps = torch.stack(cuid_pair_reps)

        combined_mention_reps = torch.cat(
            [mention_pair_reps, cuid_pair_reps[features["pair_ids"]]], dim=1
        )

        cuid_pair_logits = self.cuid_pair_classifier(cuid_pair_reps)
        mention_pair_logits = self.mention_pair_classifier(combined_mention_reps)

        if "labels" in features:
            cuid_pair_loss = self.loss(cuid_pair_logits, cuid_pair_labels)
            mention_pair_loss = self.loss(mention_pair_logits, features["labels"])

            pooled_mention_pair_logits = []
            for i, _ in enumerate(cuid_pair_labels):
                pooled_mention_pair_logits.append(torch.logsumexp(mention_pair_logits[features["pair_ids"] == i], dim=0))
            pooled_mention_pair_logits = torch.stack(pooled_mention_pair_logits)

            dist = torch.norm(torch.softmax(pooled_mention_pair_logits, dim=1) - torch.softmax(cuid_pair_logits, dim=1), 2)

            loss = cuid_pair_loss + mention_pair_loss + self.dist_penalize_factor * dist
        else:
            loss = None


        return SequenceClassifierOutput(
            loss=loss,
            logits=mention_pair_logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def collate_fn(self, batch):
        unrolled_features = []
        pair_ids = []
        collator = transformers.DataCollatorWithPadding(self.tokenizer)

        for idx_pair, pair_features_list in enumerate(batch):
            unrolled_features.extend(pair_features_list)
            pair_ids.extend([idx_pair] * len(pair_features_list))

        batched_features = collator(unrolled_features)

        batched_features["pair_ids"] = torch.tensor(pair_ids)

        return batched_features

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.lr_schedulers().step()
        self.log("train/loss", output.loss, prog_bar=False)
        preds = self.logits_to_indicators(output.logits)
        self.train_f1(preds.float(), batch["labels"].long())
        self.log("train/f1", self.train_f1, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("val/loss", output.loss, prog_bar=True)

        return {"loss": output.loss, "logits": output.logits,
                "labels": batch["labels"]}

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.cat([i["logits"] for i in outputs]).to(self.device)
        labels = torch.cat([i["labels"] for i in outputs]).long().to(self.device)
        val_f1 = torchmetrics.F1()
        indicators = self.logits_to_indicators(logits)
        val_f1(indicators.float().cpu(), labels.long().cpu())
        self.log("val/f1", val_f1, prog_bar=True)

    def configure_optimizers(self):
        assert self.num_training_steps > 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedule = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                num_warmup_steps=0.1 * self.num_training_steps,
                                                                num_training_steps=self.num_training_steps,
                                                                )

        return [optimizer], [schedule]

    def logits_to_indicators(self, logits: torch.FloatTensor) -> torch.LongTensor:
        return torch.sigmoid(logits.to(self.device)) > self.thresholds.to(self.device)

    def predict(self, collection: bioc.BioCCollection) -> None:
        f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        it = tqdm(collection.documents, desc="Predicting")
        for doc in it:
            for passage in doc.passages:
                cuid_pair_to_examples = defaultdict(list)
                for sent in passage.sentences:
                    for example in sentence_to_examples(sent, self.tokenizer):
                        cuid_pair_to_examples[(example["head_cuid"], example["tail_cuid"])].append(example)
                if cuid_pair_to_examples.values():
                    features = []
                    examples = []
                    for pair_examples in cuid_pair_to_examples.values():
                        features.append([i["features"] for i in pair_examples])
                        examples.extend(pair_examples)

                    batch = self.collate_fn(features)
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    with torch.no_grad():
                        out = self.forward(batch)
                    preds = self.logits_to_indicators(out.logits)
                    f1.update(preds.cpu().float(),
                              batch['labels'].cpu().long())
                    if "labels" in batch:
                        it.set_description(f"F1: {f1.compute():.2f}")

                    for example, preds, logits in zip(examples, preds, out.logits):
                        head = example["head"]
                        tail = example["tail"]
                        for label_idx in torch.where(preds)[0]:
                            rel = bioc.BioCRelation()
                            rel.infons["prob"] = logits[label_idx.item()]
                            rel.infons["type"] = ID_TO_LABEL[label_idx.item()]
                            rel.add_node(bioc.BioCNode(refid=head,
                                                       role="head"))
                            rel.add_node(bioc.BioCNode(refid=tail,
                                                       role="tail"))
                            passage.add_relation(rel)

        return None

    def get_dataset(self, path, limit_examples=None):
        return Dataset(path, self.tokenizer, limit_examples=limit_examples)
