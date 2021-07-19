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

log = utils.get_logger(__name__)

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
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


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


def sentence_to_examples(sentence, tokenizer, doc_context):
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

            if doc_context:
                text = insert_pair_markers(text=doc_context,
                                           head=head,
                                           tail=tail,
                                           sentence_offset=0)
            else:
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
    def __init__(self, path, tokenizer, limit_examples, use_doc_context):
        self.examples = []
        self.tokenizer = tokenizer
        with open(hydra.utils.to_absolute_path(path)) as f:
            collection = bioc.load(f)
            doc: bioc.BioCDocument
            for doc in tqdm(collection.documents, desc="Loading data"):
                if limit_examples and len(self.examples) > limit_examples:
                    break
                for passage in doc.passages:
                    for sentence in passage.sentences:
                        if use_doc_context:
                            self.examples += sentence_to_examples(sentence, tokenizer,
                                                                  doc_context=doc.passages[0].text)
                        else:
                            self.examples += sentence_to_examples(sentence, tokenizer,
                                                                  doc_context=None)



    def __getitem__(self, item):
        return self.examples[item]["features"]

    def __len__(self):
        return len(self.examples)


class EntityMarkerBaseline(pl.LightningModule):
    def __init__(self, transformer: str, lr: float, loss: str, tune_thresholds: bool,
                 use_doc_context: bool):
        super().__init__()

        loss = loss.lower().strip()
        assert loss in {"atlop", "bce", "ce"}
        if loss == "atlop":
            self.loss = ATLoss()
        elif loss == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif loss == "ce":
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError

        self.tune_thresholds = tune_thresholds
        if self.tune_thresholds and isinstance(self.loss, ATLoss):
            warnings.warn("atlop loss has no fixed thresholds. Setting tune_thresholds to False")
            self.tune_thresholds = False
        self.use_doc_context = use_doc_context

        self.transformer = transformers.AutoModel.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)
        self.tokenizer.add_tokens(["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"],
                                  special_tokens=True)
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        self.dropout = nn.Dropout(self.transformer.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.transformer.config.hidden_size * 2,
                                    len(LABEL_TO_ID))
        self.train_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        self.dev_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        self.lr = lr
        self.num_training_steps = None
        self.collate_fn = transformers.DataCollatorWithPadding(self.tokenizer)
        self.thresholds = nn.Parameter(torch.ones(len(LABEL_TO_ID)) * 0.5)

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
            labels = features["labels"]
            if isinstance(self.loss, nn.CrossEntropyLoss):
                labels = labels.argmax(dim=1)
            loss = self.loss(logits, labels)
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
        preds = self.logits_to_indicators(output.logits)
        self.train_f1(preds.float(), batch["labels"].long())
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
        if self.tune_thresholds:
            for idx_label in range(len(LABEL_TO_ID)):
                best_f1 = 0
                best_threshold = 1.0
                for idx_threshold in range(1, 100):
                    threshold = idx_threshold * 0.01
                    f1 = torchmetrics.F1(threshold=threshold)
                    label_probs = torch.sigmoid(logits[:, idx_label])
                    label_labels = labels[:, idx_label]
                    f1.update(label_probs.cpu(), label_labels.cpu())
                    if f1.compute() > best_f1:
                        best_f1 = f1.compute()
                        best_threshold = threshold
                self.thresholds[idx_label] = best_threshold
            log.info(f"Setting new thresholds: {self.thresholds})")
        indicators = self.logits_to_indicators(logits).to(self.device)
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
        if isinstance(self.loss, ATLoss):
            thresholds = logits[:, 0].unsqueeze(1)
            return logits > thresholds
        elif isinstance(self.loss, nn.BCEWithLogitsLoss):
            return torch.sigmoid(logits.to(self.device)) > self.thresholds.to(self.device)
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            indicators = logits == logits.max(dim=1)[0].unsqueeze(1)
            indicators[:, 0] = 0
            return indicators
        else:
            raise ValueError

    def predict(self, collection: bioc.BioCCollection) -> None:
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID))
        it = tqdm(collection.documents, desc="Predicting")
        for doc in it:
            for passage in doc.passages:
                for sent in passage.sentences:
                    if self.use_doc_context:
                        examples = sentence_to_examples(sent, self.tokenizer,
                                                        doc_context=doc.passages[0].text)
                    else:
                        examples = sentence_to_examples(sent, self.tokenizer,
                                                        doc_context=None)
                    if examples:
                        batch = collator([i["features"] for i in examples])
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)
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
                                sent.add_relation(rel)

        return None

    def get_dataset(self, path, limit_examples=None):
        return Dataset(path, self.tokenizer, limit_examples=limit_examples,
                       use_doc_context=self.use_doc_context)
