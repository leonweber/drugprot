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
from drugprot.metrics import SequenceAccuracy

pl.seed_everything(42)

LABEL_TO_TARGET = {
    # DrugProt
#     "NONE": "none",
#     "ACTIVATOR": "direct regulator / activator",
#     "AGONIST": "direct regulator / agonist",
#     "AGONIST-ACTIVATOR": "direct regulator / agonist / activator",
#     "AGONIST-INHIBITOR": "direct regulator / agonist / inhibitor",
#     "ANTAGONIST": "direct regulator / antagonist",
#     "DIRECT-REGULATOR": "direct regulator",
#     "INDIRECT-DOWNREGULATOR": "indirect regulator / downregulator",
#     "INDIRECT-UPREGULATOR": "indirect regulator / upregulator",
#     "INHIBITOR": "direct regulator / inhibitor",
#     "PART-OF": "part of",
#     "PRODUCT-OF": "direct regulator / product of",
#     "SUBSTRATE": "direct regulator / substrate",
#     "SUBSTRATE_PRODUCT-OF": "direct regulator / substrate / product of",
    
    "NONE": "none",
    "ACTIVATOR": "activator",
    "AGONIST": "agonist",
    "AGONIST-ACTIVATOR": "agonist-activator",
    "AGONIST-INHIBITOR": "agonist-inhibitor",
    "ANTAGONIST": "antagonist",
    "DIRECT-REGULATOR": "direct-regulator",
    "INDIRECT-DOWNREGULATOR": "indirect-downregulator",
    "INDIRECT-UPREGULATOR": "indirect-upregulator",
    "INHIBITOR": "inhibitor",
    "PART-OF": "part-of",
    "PRODUCT-OF": "product-of",
    "SUBSTRATE": "substrate",
    "SUBSTRATE_PRODUCT-OF": "substrate product-of",
    
    "COFACTOR": "cofactor",
    "DOWNREGULATOR": "downregulator",
    "INDIRECT-REGULATOR": "indirect-regulator",
    "MODULATOR": "modulator",
    "MODULATOR-ACTIVATOR": "modulator-activator",
    "MODULATOR-INHIBITOR": "modulator-inhibitor",
    "NOT": "not",
    "REGULATOR": "regulator",
    "UNDEFINED": "undefined",
    "UPREGULATOR": "upregulator",
    
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

TARGET_TO_LABEL = {v: k for k, v in LABEL_TO_TARGET.items()}


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
                                             insertion="** ",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=ends[0],
                                             text=text,
                                             insertion=" **",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=starts[1],
                                             text=text,
                                             insertion="## ",
                                             starts=starts,
                                             ends=ends
                                             )
    text, starts, ends = insert_consistently(offset=ends[1],
                                             text=text,
                                             insertion=" ##",
                                             starts=starts,
                                             ends=ends
                                             )

    def find_1st(string, substring):
        return string.find(substring)
    
    def find_2nd(string, substring):
        return string.find(substring, string.find(substring) + 1)
    
    marked_head = text[find_1st(text, "**"):find_2nd(text, "**")].replace("**", "").replace("##", "").strip()
    marked_tail = text[find_1st(text, "##"):find_2nd(text, "##")].replace("**", "").replace("##", "").strip()
    
    #print("marked_head: ", marked_head)
    #print("marked_tail: ", marked_tail)
    #assert (marked_head == head.text or not head.text.isalnum()), f"{text}" # skip unicode errors
    #assert (marked_tail == tail.text or not tail.text.isalnum()), f"{text}" # skip unicode errors

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

            #print("text: ", text)
            features = tokenizer(text, max_length=512, truncation=True)

            try:
                assert "##" in tokenizer.decode(features["input_ids"])
                assert "**" in tokenizer.decode(features["input_ids"])
            except AssertionError:
                continue # entity was truncated

            labels = []
            for label in pair_to_relations[(head.id, tail.id)]:
                if label in LABEL_TO_TARGET:
                    labels.append(LABEL_TO_TARGET[label])
            #print("labels: ", "|".join(labels))
            tokenizer_output = tokenizer("|".join(labels), max_length=15, padding="max_length")
            features["labels"] = tokenizer_output["input_ids"]
            features["decoder_attention_mask"] = tokenizer_output["attention_mask"]

            examples.append(
                {"head": head.id, "tail": tail.id, "features": features}
            )

    #print("examples: ", examples)
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


class Seq2SeqBaseline(pl.LightningModule):
    def __init__(self, transformer: str, lr: float):
        super().__init__()
        self.transformer = transformers.AutoModelForSeq2SeqLM.from_pretrained(transformer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)

        #self.train_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID), threshold=0.5)
        #self.dev_f1 = torchmetrics.F1(num_classes=len(LABEL_TO_ID), threshold=0.5)
        self.train_seq_acc = SequenceAccuracy()
        self.val_seq_acc = SequenceAccuracy()
        self.lr = lr
        self.num_training_steps = None
        self.collate_fn = transformers.DataCollatorWithPadding(self.tokenizer)

    def forward(self, features):
        labels = None
        if "labels" in features:
            labels = features["labels"]
            labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
            features["labels"] = labels
        
        #print("features: ", features)
        output = self.transformer(input_ids=features["input_ids"],
                                  attention_mask=features["attention_mask"],
                                  decoder_input_ids=features.get("decoder_input_ids"),
                                  decoder_attention_mask=features.get("decoder_attention_mask"),
                                  labels=features.get("labels"),
                                  )
        #print("loss: ", output.loss)
        #print("logits: ", output.logits)
        return output

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.lr_schedulers().step()
        self.log("train/loss", output.loss, prog_bar=False)
        self.train_seq_acc(torch.argmax(output.logits, dim=-1), batch["labels"].long())
        self.log("train/seq_accuracy", self.train_seq_acc, prog_bar=True)
        #self.train_f1(torch.sigmoid(output.logits), batch["labels"].long())
        #self.log("train/f1", self.train_f1, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        self.log("val/loss", output.loss, prog_bar=True)
        self.val_seq_acc(torch.argmax(output.logits, dim=-1), batch["labels"].long())
        self.log("val/seq_accuracy", self.val_seq_acc, prog_bar=True)
        #self.dev_f1(torch.sigmoid(output.logits), batch["labels"].long())
        #self.log("val/f1", self.dev_f1, prog_bar=True)

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
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        it = tqdm(collection.documents, desc="Predicting")
        for doc in it:
            for passage in doc.passages:
                for sent in passage.sentences:
                    examples = sentence_to_examples(sent, self.tokenizer)
                    if examples:
                        batch = collator([i["features"] for i in examples])
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)
                        
                        outputs = self.transformer.generate(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            max_length=15,
                            early_stopping=True
                        )

                        for example, output in zip(examples, outputs):
                            head = example["head"]
                            tail = example["tail"]
                            line = self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            if not line:
                                continue
                            
                            for label_str in line.split("|"):
                                rel = bioc.BioCRelation()
                                rel.infons["prob"] = 1.0
                                
                                if label_str not in TARGET_TO_LABEL:
                                    print("Unknown label: ", label_str)
                                    continue
                                
                                rel.infons["type"] = TARGET_TO_LABEL[label_str]
                                rel.add_node(bioc.BioCNode(refid=head,
                                                           role="head"))
                                rel.add_node(bioc.BioCNode(refid=tail,
                                                           role="tail"))
                                sent.add_relation(rel)

        return None


    def get_dataset(self, path, limit_examples=None):
        return Dataset(path, self.tokenizer, limit_examples=limit_examples)
