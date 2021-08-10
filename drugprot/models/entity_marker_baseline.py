import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra.utils
import numpy as np
import torch
import pytorch_lightning as pl
import bioc
import torchmetrics
import transformers
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.file_utils import ModelOutput
import torch.nn.functional as F

from drugprot import utils

log = utils.get_logger(__name__)


def overlaps(a, b):
    a = [int(i) for i in a]
    b = [int(i) for i in b]
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


@dataclass
class MultitaskClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    dataset_to_logits: Dict[str, torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def sentence_to_examples(
    sentence, tokenizer, doc_context, pair_types, mark_with_special_tokens, blind_entities, label_to_id=None, max_length=512, use_none_class=False
):
    examples = []

    ann_id_to_ann = {}
    for ann in sentence.annotations:
        ann_id_to_ann[ann.id] = ann

    pair_to_relations = defaultdict(set)
    for rel in sentence.relations:
        head = rel.get_node("head").refid
        tail = rel.get_node("tail").refid
        rel = rel.infons["type"]
        pair_to_relations[(head, tail)].add(rel)

    for head in sentence.annotations:
        for tail in sentence.annotations:
            if (
                pair_types
                and (head.infons["type"], tail.infons["type"]) not in pair_types
            ):
                continue
            if head.id == tail.id:
                continue
            if doc_context:
                text = insert_pair_markers(
                    text=doc_context, head=head, tail=tail, sentence_offset=0, mark_with_special_tokens=mark_with_special_tokens, blind_entities=blind_entities
                )
            else:
                text = insert_pair_markers(
                    text=sentence.text,
                    head=head,
                    tail=tail,
                    sentence_offset=sentence.offset,
                    mark_with_special_tokens=mark_with_special_tokens, blind_entities=blind_entities
                )

            features = tokenizer.encode_plus(
                text, max_length=max_length, truncation=True
            )

            if mark_with_special_tokens:
                try:
                    assert "HEAD-S" in tokenizer.decode(features["input_ids"])
                    assert "HEAD-E" in tokenizer.decode(features["input_ids"])
                    assert "TAIL-S" in tokenizer.decode(features["input_ids"])
                    assert "TAIL-E" in tokenizer.decode(features["input_ids"])
                except AssertionError:
                    log.warning("Truncated entity")
                    continue  # entity was truncated

            if label_to_id:
                features["labels"] = np.zeros(len(label_to_id))
                for label in pair_to_relations[(head.id, tail.id)]:
                    features["labels"][label_to_id[label]] = 1

                if use_none_class and features["labels"].sum() == 0:
                    features["labels"][0] = 1

            examples.append({"head": head.id, "tail": tail.id, "features": features})

    return examples


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = logits > th_logit
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.0).to(logits)
        return output


def insert_consistently(offset, insertion, text, starts, ends):
    new_text = text[:offset] + insertion + text[offset:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[starts >= offset] += len(insertion)
    new_ends[ends >= offset] += len(insertion)

    return new_text, new_starts, new_ends


def delete_consistently(from_idx, to_idx , text, starts, ends):
    assert to_idx >= from_idx
    new_text = text[:from_idx] + text[to_idx:]
    new_starts = starts.copy()
    new_ends = ends.copy()
    new_starts[(from_idx <= starts) & (starts <= to_idx)] = from_idx
    new_ends[(from_idx <= ends) & (ends <= to_idx)] = from_idx
    new_starts[starts > to_idx] -= (to_idx - from_idx)
    new_ends[ends > to_idx] -= (to_idx - from_idx)

    return new_text, new_starts, new_ends


def insert_pair_markers(text, head, tail, sentence_offset, mark_with_special_tokens, blind_entities):
    if mark_with_special_tokens:
        head_start_marker = "[HEAD-S]"
        tail_start_marker = "[TAIL-S]"
        head_end_marker = "[HEAD-E]"
        tail_end_marker = "[TAIL-E]"
    else:
        head_start_marker = "@"
        tail_start_marker = "@"
        head_end_marker = "$"
        tail_end_marker = "$"


    starts = (
        np.array([head.locations[0].offset, tail.locations[0].offset]) - sentence_offset
    )
    ends = starts + [head.locations[0].length, tail.locations[0].length]


    if blind_entities:
        span_e1 = (starts[0], ends[0])
        span_e2 = (starts[1], ends[1])
        if overlaps(span_e1, span_e2):
            delete_span = (starts.min(), ends.max())
            text, starts, ends = delete_consistently(from_idx=delete_span[0], to_idx=delete_span[1], text=text, starts=starts, ends=ends)
            text, starts, ends = insert_consistently(offset=starts[0], text=text, insertion="HEAD-TAIL", starts=starts, ends=ends)
            starts -= len("HEAD-TAIL")
            ends[:] = starts + len("HEAD-TAIL")
        else:
            text, starts, ends = delete_consistently(from_idx=starts[0], to_idx=ends[0], text=text, starts=starts, ends=ends)
            text, starts, ends = insert_consistently(offset=starts[0], text=text, insertion="HEAD", starts=starts, ends=ends)
            starts[0] -= len("HEAD")
            ends[0] = starts[0] + len("HEAD")

            text, starts, ends = delete_consistently(from_idx=starts[1], to_idx=ends[1], text=text, starts=starts, ends=ends)
            text, starts, ends = insert_consistently(offset=starts[1], text=text, insertion="TAIL", starts=starts, ends=ends)
            starts[1] -= len("TAIL")
            ends[1] = starts[1] + len("TAIL")

    text, starts, ends = insert_consistently(
        offset=starts[0], text=text, insertion=head_start_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=ends[0], text=text, insertion=head_end_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=starts[1], text=text, insertion=tail_start_marker, starts=starts, ends=ends
    )
    text, starts, ends = insert_consistently(
        offset=ends[1], text=text, insertion=tail_end_marker, starts=starts, ends=ends
    )
    # marked_head = (
    #     text[text.index("[HEAD-S]") : text.index("[HEAD-E]")]
    #     .replace("[HEAD-S]", "")
    #     .replace("[HEAD-E]", "")
    #     .replace("[TAIL-S]", "")
    #     .replace("[TAIL-E]", "")
    # )
    # marked_tail = (
    #     text[text.index("[TAIL-S]") : text.index("[TAIL-E]")]
    #     .replace("[HEAD-S]", "")
    #     .replace("[HEAD-E]", "")
    #     .replace("[TAIL-S]", "")
    #     .replace("[TAIL-E]", "")
    # )
    # assert (
    #     marked_head == head.text or not head.text.isalnum()
    # )  # skip unicode errors and disjoint entities from ddi
    # assert (
    #     marked_tail == tail.text or not tail.text.isalnum()
    # )  # skip unicode errors and disjoint entities from ddi

    return text


class BiocDataset:
    def __init__(self, path, tokenizer, limit_examples, use_doc_context,
                 mark_with_special_tokens, blind_entities, max_length,
                 use_none_class=False):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta = utils.get_dataset_metadata(path)
        self.use_none_class = use_none_class
        with open(hydra.utils.to_absolute_path(path)) as f:
            collection = bioc.load(f)
            doc: bioc.BioCDocument
            for doc in tqdm(collection.documents, desc="Loading data"):
                if limit_examples and len(self.examples) > limit_examples:
                    break
                for passage in doc.passages:
                    for sentence in passage.sentences:
                        if use_doc_context:
                            doc_context = doc.passages[0].text
                        else:
                            doc_context = None
                        self.examples += sentence_to_examples(
                            sentence,
                            tokenizer,
                            doc_context=doc_context,
                            pair_types=self.meta.pair_types,
                            label_to_id=self.meta.label_to_id,
                            mark_with_special_tokens=mark_with_special_tokens,
                            blind_entities=blind_entities,
                            max_length=self.max_length,
                            use_none_class=use_none_class
                        )

                        if limit_examples and len(self.examples) > limit_examples:
                            break

    def __getitem__(self, item):
        example = self.examples[item]["features"].copy()
        example["meta"] = self.meta

        return example

    def __len__(self):
        return len(self.examples)


class EntityMarkerBaseline(pl.LightningModule):
    def __init__(
        self,
        transformer: str,
        lr: float,
        finetune_lr: float,
        loss: str,
        tune_thresholds: bool,
        use_doc_context: bool,
        dataset_to_meta: Dict[str, utils.DatasetMetaInformation],
        max_length: int,
        optimized_metric: str,
        use_cls: bool = False,
        use_starts: bool = False,
        use_ends: bool = False,
        mark_with_special_tokens: bool = True,
        blind_entities: bool = True,
    ):
        super().__init__()

        self.use_cls = use_cls
        self.max_length = max_length
        self.optimized_metric = optimized_metric
        self.use_starts = use_starts
        self.use_ends = use_ends
        self.mark_with_special_tokens = mark_with_special_tokens
        self.blind_entities = blind_entities

        assert use_cls or use_starts or use_ends

        if not mark_with_special_tokens:
            assert (
                not use_starts and not use_ends
            ), "Starts and ends cannot be uniquely determined without additional special tokens"

        self.dataset_to_meta = dataset_to_meta
        loss = loss.lower().strip()
        assert loss in {"atlop", "bce"}
        if loss == "atlop":
            self.loss = ATLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

        self.tune_thresholds = tune_thresholds
        if self.tune_thresholds and isinstance(self.loss, ATLoss):
            warnings.warn(
                "atlop loss has no fixed thresholds. Setting tune_thresholds to False"
            )
            self.tune_thresholds = False
        self.use_doc_context = use_doc_context

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)

        if mark_with_special_tokens:
            self.tokenizer.add_tokens(
                ["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"], special_tokens=True
            )
            self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)

        self.dataset_to_classifier = nn.ModuleDict()
        self.dataset_to_train_f1 = {}
        self.dataset_to_dev_f1 = {}
        seq_rep_size = 0
        if use_cls:
            seq_rep_size += self.transformer.config.hidden_size
        if use_starts:
            seq_rep_size += 2 * self.transformer.config.hidden_size
        if use_ends:
            seq_rep_size += 2 * self.transformer.config.hidden_size
        for dataset, meta in dataset_to_meta.items():
            self.dataset_to_classifier[dataset] = nn.Linear(
                seq_rep_size, len(meta.label_to_id)
            )
            self.dataset_to_train_f1[dataset] = torchmetrics.F1(
                num_classes=len(meta.label_to_id)
            )
            self.dataset_to_dev_f1[dataset] = torchmetrics.F1(
                num_classes=len(meta.label_to_id)
            )

        self.lr = lr
        self.finetune_lr = finetune_lr
        self.num_training_steps = None

    def collate_fn(self, data):
        meta = data[0].pop("meta")
        for i in data[1:]:
            m = i.pop("meta")
            assert m == meta
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        batch = collator(data)
        batch["meta"] = meta

        return batch

    def forward(self, features):
        output = self.transformer(
            input_ids=features["input_ids"],
            token_type_ids=features["token_type_ids"],
            attention_mask=features["attention_mask"],
        )
        seq_emb = self.dropout(output.last_hidden_state)

        seq_reps = []

        if self.use_cls:
            seq_reps.append(seq_emb[:, 0])

        if self.use_starts:
            head_start_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids("[HEAD-S]")
            )
            tail_start_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids("[TAIL-S]")
            )
            head_start_rep = seq_emb[head_start_idx]
            tail_start_rep = seq_emb[tail_start_idx]
            start_pair_rep = torch.cat([head_start_rep, tail_start_rep], dim=1)
            seq_reps.append(start_pair_rep)

        if self.use_ends:
            head_end_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids("[HEAD-E]")
            )
            tail_end_idx = torch.where(
                features["input_ids"]
                == self.tokenizer.convert_tokens_to_ids("[TAIL-E]")
            )
            head_end_rep = seq_emb[head_end_idx]
            tail_end_rep = seq_emb[tail_end_idx]
            end_pair_rep = torch.cat([head_end_rep, tail_end_rep], dim=1)
            seq_reps.append(end_pair_rep)

        seq_reps = torch.cat(seq_reps, dim=1)
        dataset_to_logits = {}

        for dataset, classifier in self.dataset_to_classifier.items():
            dataset_to_logits[dataset] = classifier(seq_reps)
        if "labels" in features:
            logits = dataset_to_logits[features["meta"].name]
            loss = self.loss(logits, features["labels"])
        else:
            loss = None

        return MultitaskClassifierOutput(
            loss=loss,
            dataset_to_logits=dataset_to_logits,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def training_step(self, batch, batch_idx):
        dataset = batch["meta"].name
        output = self.forward(batch)
        self.lr_schedulers().step()

        self.log("total/train/loss", output.loss, prog_bar=True)
        train_f1 = self.dataset_to_dev_f1[dataset]
        indicators = self.logits_to_indicators(output.dataset_to_logits[dataset]).to(
            self.device
        )
        train_f1(indicators.float().cpu(), batch["labels"].long().cpu())
        self.log(f"{dataset}/train/loss", output.loss, prog_bar=False)
        if dataset == "drugprot":
            self.log(f"{dataset}/train/f1", train_f1, prog_bar=True)
        else:
            self.log(f"{dataset}/train/f1", train_f1, prog_bar=False)
        return output.loss

    def validation_step(self, batch, batch_idx):
        dataset = batch["meta"].name
        output = self.forward(batch)
        self.log("total/val/loss", output.loss, prog_bar=True)
        val_f1 = self.dataset_to_dev_f1[dataset]
        indicators = self.logits_to_indicators(output.dataset_to_logits[dataset]).to(
            self.device
        )
        val_f1(indicators.float().cpu(), batch["labels"].long().cpu())
        self.log(f"{dataset}/val/loss", output.loss, prog_bar=False)
        if dataset == "drugprot":
            self.log(f"{dataset}/val/f1", val_f1, prog_bar=True)
        else:
            self.log(f"{dataset}/val/f1", val_f1, prog_bar=False)

        return output.loss

    def configure_optimizers(self):
        assert self.num_training_steps > 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        schedule = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * self.num_training_steps,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [schedule]

    def logits_to_indicators(self, logits: torch.FloatTensor) -> torch.LongTensor:
        if isinstance(self.loss, ATLoss):
            thresholds = logits[:, 0].unsqueeze(1)
            return logits > thresholds
        elif isinstance(self.loss, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits.to(self.device)) > 0.5
        else:
            raise ValueError

    def predict(self, collection: bioc.BioCCollection) -> None:
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        it = tqdm(collection.documents, desc="Predicting")
        for doc in it:
            for dataset, meta in self.dataset_to_meta.items():
                for passage in doc.passages:
                    for sent in passage.sentences:
                        if self.use_doc_context:
                            examples = sentence_to_examples(
                                sent,
                                self.tokenizer,
                                doc_context=doc.passages[0].text,
                                pair_types=meta.pair_types,
                                label_to_id=None,
                                max_length=self.max_length,
                                mark_with_special_tokens=self.mark_with_special_tokens,
                                blind_entities=self.blind_entities
                            )
                        else:
                            examples = sentence_to_examples(
                                sent,
                                self.tokenizer,
                                doc_context=None,
                                pair_types=meta.pair_types,
                                label_to_id=None,
                                max_length=self.max_length,
                                mark_with_special_tokens=self.mark_with_special_tokens,
                                blind_entities=self.blind_entities
                            )
                        if examples:
                            batch = collator([i["features"] for i in examples])
                            for k, v in batch.items():
                                batch[k] = v.to(self.device)
                            out = self.forward(batch)
                            dataset_logits = out.dataset_to_logits[dataset]
                            preds = self.logits_to_indicators(dataset_logits)
                            for example, preds, logits in zip(
                                examples, preds, dataset_logits
                            ):
                                head = example["head"]
                                tail = example["tail"]
                                for label_idx in torch.where(preds)[0]:
                                    rel = bioc.BioCRelation()
                                    rel.infons["prob"] = logits[label_idx.item()]
                                    rel.infons["type"] = (
                                        dataset
                                        + "/"
                                        + self.dataset_to_meta[dataset]["id_to_label"][
                                            label_idx.item()
                                        ]
                                    )
                                    rel.add_node(bioc.BioCNode(refid=head, role="head"))
                                    rel.add_node(bioc.BioCNode(refid=tail, role="tail"))
                                    sent.add_relation(rel)

        return None

    def get_dataset(self, path, limit_examples=None):
        return BiocDataset(path, self.tokenizer, limit_examples=limit_examples,
                           use_doc_context=self.use_doc_context,
                           mark_with_special_tokens=self.mark_with_special_tokens,
                           blind_entities=self.blind_entities,
                           max_length=self.max_length)
