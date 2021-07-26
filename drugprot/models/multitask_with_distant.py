import abc
import random
from collections import defaultdict
from typing import Dict, Optional, Tuple

import hydra.utils
import numpy as np
import torch
import pytorch_lightning as pl
import bioc
import torchmetrics
import transformers
from torch import nn
from tqdm import tqdm

from drugprot import utils
from drugprot.models.entity_marker_baseline import BiocDataset, \
    MultitaskClassifierOutput, sentence_to_examples
from drugprot.utils import DatasetMetaInformation

log = utils.get_logger(__name__)


class DistantDataset:
    def __init__(self, path, tokenizer, limit_examples, max_length, max_examples_per_pair=20):
        self.pair_to_examples = defaultdict(list)
        self.tokenizer = tokenizer
        self.meta = utils.get_dataset_metadata(path)
        self.max_length = max_length
        self.max_examples_per_pair = max_examples_per_pair
        with open(hydra.utils.to_absolute_path(path)) as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Loading data"):
                if limit_examples and len(self.pair_to_examples) > limit_examples:
                    break

                fields = line.strip().split("\t")
                if len(fields) < 2:
                    continue

                head_type, head_cuid, tail_type, tail_cuid, relations, text, pmid = fields
                relations = relations.split(",")
                pair = ((head_type.upper(), head_cuid), (tail_type.upper(), tail_cuid))

                features = tokenizer.encode_plus(text, max_length=self.max_length,
                                                 truncation=True)
                labels = np.zeros(len(self.meta.label_to_id))
                for rel in relations:
                    labels[self.meta.label_to_id[rel]] = 1

                features["labels"] = labels

                self.pair_to_examples[pair].append(features)

        self.pairs = sorted(self.pair_to_examples.keys())

    def __getitem__(self, item):
        examples = self.pair_to_examples[self.pairs[item]]
        if len(examples) > self.max_examples_per_pair:
            examples = random.sample(examples, self.max_examples_per_pair)
        # while len(examples) < self.max_examples_per_pair:
        #     examples.append(examples[0].copy())

        for example in examples:
            example["meta"] = self.meta
        return examples

    def __len__(self):
        return len(self.pairs)


class Aggregator(abc.ABC, nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    @abc.abstractmethod
    def forward(self, seq_rep: torch.FloatTensor, features: Dict) -> torch.FloatTensor:
        pass


class LSEAggregator(Aggregator):
    def forward(self, input: Tuple) -> torch.FloatTensor:
        pair_tensors = []

        seq_rep, features = input
        for pair_id in features["pair_ids"].unique():
            pair_tensors.append(torch.logsumexp(seq_rep[features["pair_ids"] == pair_id], dim=0))

        return torch.stack(pair_tensors, dim=0)


class IdentityAggregator(nn.Module):
    def forward(self, input: Tuple) -> torch.FloatTensor:
        return input[0]


class MultitaskWithDistantModel(pl.LightningModule):
    def __init__(
        self,
        transformer: str,
        aggregator: str,
        lr: float,
        finetune_lr: float,
        use_doc_context: bool,
        sentence_batch_size: int,
        distant_batch_size: int,
        dataset_to_meta: Dict[str, DatasetMetaInformation],
        optimized_metric: str,
        max_length,
        use_cls: bool = False,
        use_starts: bool = False,
        use_ends: bool = False,
        blind_entities: bool = False,
        dropout=0.3):
        super().__init__()

        self.use_cls = use_cls
        self.use_starts = use_starts
        self.use_ends = use_ends
        self.blind_entities = blind_entities
        self.sentence_batch_size = sentence_batch_size
        self.distant_batch_size = distant_batch_size
        self.max_length = max_length
        self.optimized_metric = optimized_metric

        assert use_cls or use_starts or use_ends


        self.dataset_to_meta = dataset_to_meta
        self.loss = nn.BCEWithLogitsLoss()

        self.use_doc_context = use_doc_context

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer,
                                                                    use_fast=True)

        self.tokenizer.add_tokens(
            ["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"], special_tokens=True
        )
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.transformer_dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.non_transformer_dropout = nn.Dropout(dropout)

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

        assert aggregator in {"lse", "max", "mean", "attention"}
        if aggregator == "lse":
            self.aggregator = LSEAggregator(self.transformer.config.hidden_size)
        else:
            raise NotImplementedError

        for dataset, meta in dataset_to_meta.items():
            out_layer = self.dataset_to_classifier[dataset] = nn.Linear(
                seq_rep_size, len(meta.label_to_id)
            )
            if meta.type == "distant":
                classifier = nn.Sequential(self.aggregator,
                                           self.non_transformer_dropout,
                                           out_layer)
            elif meta.type == "sentence":
                classifier = nn.Sequential(IdentityAggregator(), out_layer)
            else:
                raise ValueError(meta.type)
            self.dataset_to_classifier[dataset] = classifier
            self.dataset_to_train_f1[dataset] = torchmetrics.F1(
                num_classes=len(meta.label_to_id)
            )
            self.dataset_to_dev_f1[dataset] = torchmetrics.F1(
                num_classes=len(meta.label_to_id)
            )

        self.lr = lr
        self.finetune_lr = finetune_lr
        self.num_training_steps = None

    def mil_collate_fn(self, batch):
        unrolled_features = []
        pair_ids = []
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        metas = []
        labels = []

        for idx_pair, pair_features_list in enumerate(batch):
            unrolled_features.extend(pair_features_list)
            pair_ids.extend([idx_pair] * len(pair_features_list))
            metas.extend(i.pop("meta") for i in pair_features_list)
            if "labels" in pair_features_list[0]:
                labels.append(torch.tensor(pair_features_list[0]["labels"]))

        batched_features = collator(unrolled_features)

        batched_features["pair_ids"] = torch.tensor(pair_ids)

        assert all(meta == metas[0] for meta in metas)
        batched_features["meta"] = metas[0]

        if labels:
            batched_features["labels"] = torch.stack(labels, dim=0)

        return batched_features

    def collate_fn(self, data):
        meta: DatasetMetaInformation
        try:
            meta = data[0].pop("meta")
            collator = transformers.DataCollatorWithPadding(self.tokenizer)
            for i in data[1:]:
                m = i.pop("meta")
                assert m == meta
            batch = collator(data)
            batch["meta"] = meta
        except TypeError:
            batch = self.mil_collate_fn(data)

        return batch

    def forward(self, features):
        output = self.transformer(
            input_ids=features["input_ids"],
            token_type_ids=features["token_type_ids"],
            attention_mask=features["attention_mask"],
        )
        seq_emb = self.transformer_dropout(output.last_hidden_state)

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
            if features["meta"].type == "sentence" and not isinstance(list(classifier.modules())[1], IdentityAggregator): # MIL for sentence batches makes no sense
                continue
            dataset_to_logits[dataset] = classifier((seq_reps, features))
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
        dataset = batch.meta.name
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
        return torch.sigmoid(logits.to(self.device)) > 0.5

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
                                max_length=512,
                                blind_entities=self.blind_entities,
                                mark_with_special_tokens=True
                            )
                        else:
                            examples = sentence_to_examples(
                                sent,
                                self.tokenizer,
                                doc_context=None,
                                pair_types=meta.pair_types,
                                label_to_id=None,
                                max_length=512,
                                blind_entities=self.blind_entities,
                                mark_with_special_tokens=True
                            )
                        if examples:
                            batch = collator([i["features"] for i in examples])
                            batch["meta"] = self.dataset_to_meta["drugprot"]
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
                                        + self.dataset_to_meta[dataset].id_to_label[
                                            label_idx.item()
                                        ]
                                    )
                                    rel.add_node(bioc.BioCNode(refid=head, role="head"))
                                    rel.add_node(bioc.BioCNode(refid=tail, role="tail"))
                                    sent.add_relation(rel)

        return None

    def get_dataset(self, path, limit_examples=None):
        meta = utils.get_dataset_metadata(path)
        if meta.type == "distant":
            return DistantDataset(
                path,
                self.tokenizer,
                max_length=self.max_length,
                limit_examples=limit_examples,
            )
        elif meta.type == "sentence":
            return BiocDataset(
                path,
                self.tokenizer,
                limit_examples=limit_examples,
                use_doc_context=self.use_doc_context,
                mark_with_special_tokens=True,
                blind_entities=self.blind_entities,
                max_length=self.max_length
            )
        else:
            raise ValueError(meta.type)
