import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import numpy as np

from pandas import DataFrame
from typing import List, Dict, Optional
from torch import Tensor
from torch.utils.data import Dataset

from drugprot import utils

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
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class EntityInteractionDataSet(Dataset):

    def __init__(self, data: DataFrame, entity_dict: Dict[str, int], use_unk: bool, use_none: bool):
        self.examples = self.data_to_examples(data, entity_dict, use_unk, use_none)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def data_to_examples(data: DataFrame, entity_dict: Dict[str, int],  use_unk: bool, use_none:bool):
        examples = []

        num_missing_heads = 0
        num_missing_tails = 0

        for _, row in data.iterrows():
            head_entity = row["head"]
            if head_entity not in entity_dict:
                num_missing_heads += 1
                head_entity = "DRUG-UNK" if use_unk else None

            tail_entity = row["tail"]
            if tail_entity not in entity_dict:
                num_missing_tails += 1
                tail_entity = "GENE-UNK" if use_unk else None

            if head_entity is None or tail_entity is None:
                continue

            head = entity_dict[head_entity]
            tail = entity_dict[tail_entity]

            relations_enc = np.zeros(len(LABEL_TO_ID))
            relations = row["relations"]
            if type(relations) == float: # Strange pandas issue - nan will be type float
                relations = ["NONE"] if use_none else []
            else:
                relations = relations.split("|")

            for relation in relations:
                relations_enc[LABEL_TO_ID[relation]] = 1.0

            examples.append({
                "head": torch.tensor(head),
                "tail": torch.tensor(tail),
                "labels": relations_enc
            })

        log.info(f"Data set size: {len(examples)}")
        log.info(f"Missing heads: {num_missing_heads}")
        log.info(f"Missing tails: {num_missing_tails}")

        return examples


class ClassifierOutput:

    def __init__(self, logits: torch.FloatTensor, loss: Optional[torch.FloatTensor] = None):
        self.logits = logits
        self.loss = loss


class MultiLabelClassificationNetwork(pl.LightningModule):

    def __init__(self, embeddings: Tensor, hidden_sizes: List[int], activation: Optional[str],
                 emb_trainable: bool, lr: float, l1_lambda: float):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            _weight=embeddings,
        )
        self.embedding.weight.requires_grad = emb_trainable

        prev_output_size = 2*embeddings.shape[1]
        layers = []
        for hidden_size in hidden_sizes:
            layers += [nn.Linear(prev_output_size, hidden_size)]
            if activation:
                layers += [ self.get_activation_layer(activation) ]
            prev_output_size = hidden_size

        layers += [nn.Linear(prev_output_size, len(LABEL_TO_ID))]

        self.cl_modul = nn.Sequential(*layers)
        self.loss = nn.BCEWithLogitsLoss()
        self.learning_rate = lr
        self.l1_lambda = l1_lambda

        self.train_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        self.dev_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))

        self.thresholds = nn.Parameter(torch.ones(len(LABEL_TO_ID)) * 0.5)

    def forward(self, batch):
        heads = batch["head"]
        tails = batch["tail"]

        head_embeddings = self.embedding(heads)
        tail_embeddings = self.embedding(tails)

        cl_input = torch.cat([head_embeddings, tail_embeddings], dim=1)
        logits = self.cl_modul.forward(cl_input)

        if "labels" in batch:
            loss = self.loss(logits, batch["labels"])

            if self.l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in self.parameters())
                loss = loss + self.l1_lambda * l1_norm
        else:
            loss = None

        return ClassifierOutput(logits=logits, loss=loss)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("train/loss", output.loss, prog_bar=False)

        predictions = self.logits_to_indicators(output.logits)
        self.train_f1(predictions.float(), batch["labels"].long())
        self.log("train/f1", self.train_f1, prog_bar=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.log("val/loss", output.loss, prog_bar=True)

        return {"loss": output.loss, "logits": output.logits, "labels": batch["labels"]}

    def validation_epoch_end(self, outputs) -> None:
        logits = torch.cat([i["logits"] for i in outputs]).to(self.device)
        labels = torch.cat([i["labels"] for i in outputs]).long().to(self.device)
        val_f1 = torchmetrics.F1()

        # if self.tune_thresholds:
        #     for idx_label in range(len(LABEL_TO_ID)):
        #         best_f1 = 0
        #         best_threshold = 1.0
        #         for idx_threshold in range(1, 100):
        #             threshold = idx_threshold * 0.01
        #             f1 = torchmetrics.F1(threshold=threshold)
        #             label_probs = torch.sigmoid(logits[:, idx_label])
        #             label_labels = labels[:, idx_label]
        #             f1.update(label_probs.cpu(), label_labels.cpu())
        #             if f1.compute() > best_f1:
        #                 best_f1 = f1.compute()
        #                 best_threshold = threshold
        #         self.thresholds[idx_label] = best_threshold
        #     log.info(f"Setting new thresholds: {self.thresholds})")

        indicators = self.logits_to_indicators(logits).to(self.device)
        val_f1(indicators.float().cpu(), labels.long().cpu())
        self.log("val/f1", val_f1, prog_bar=True)

    def configure_optimizers(self):
        assert self.num_training_steps > 0
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def logits_to_indicators(self, logits: torch.FloatTensor):
        # if isinstance(self.loss, ATLoss):
        #     thresholds = logits[:, 0].unsqueeze(1)
        #     return logits > thresholds

        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits.to(self.device)) > self.thresholds.to(self.device)
        else:
            raise ValueError

    @staticmethod
    def get_activation_layer(name: str):
        if name == "tanh":
            return nn.Tanh()
        elif name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "selu":
            return nn.SELU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "elu":
            return nn.ELU()
        elif name == "lrelu":
            return nn.LeakyReLU()
        else:
            raise NotImplementedError(f"Unsupported activation {name}")
