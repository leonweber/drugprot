from typing import Dict

import torch
import pytorch_lightning as pl
import bioc
import torchmetrics
import transformers
from tqdm import tqdm

from drugprot import utils
from drugprot.explain.BERT import ExplanationGenerator
from drugprot.explain.BERT.BertForSequenceClassification import BertForSequenceClassification
from drugprot.models.entity_marker_baseline import BiocDataset, sentence_to_examples

log = utils.get_logger(__name__)


class ExplainableBert(pl.LightningModule):
    def __init__(
            self,
            transformer: str,
            lr: float,
            finetune_lr: float,
            dataset_to_meta: Dict[str, utils.DatasetMetaInformation],
            max_length: int,
            optimized_metric: str,
            use_cls: bool = False,
    ):
        super().__init__()

        self.use_cls = use_cls
        self.max_length = max_length
        self.optimized_metric = optimized_metric

        assert len(dataset_to_meta) == 1
        self.meta = list(dataset_to_meta.values())[0]

        self.transformer = BertForSequenceClassification.from_pretrained(transformer, num_labels=len(self.meta.label_to_id))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(transformer)

        self.tokenizer.add_tokens(
            ["[HEAD-S]", "[HEAD-E]", "[TAIL-S]", "[TAIL-E]"], special_tokens=True
        )
        self.transformer.resize_token_embeddings(len(self.tokenizer))

        self.train_f1 = torchmetrics.F1(
                num_classes=len(self.meta.label_to_id)
            )
        self.val_f1 = torchmetrics.F1(
            num_classes=len(self.meta.label_to_id)
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
        labels = features.get("labels")
        if labels is not None:
            labels = labels.argmax(dim=1)
        output = self.transformer(
            input_ids=features["input_ids"],
            token_type_ids=features["token_type_ids"],
            attention_mask=features["attention_mask"],
            labels=labels
        )

        return output

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        self.lr_schedulers().step()

        self.log("total/train/loss", output.loss, prog_bar=True)
        indicators = self.logits_to_indicators(output.logits).to(
            self.device
        )
        self.train_f1(indicators.float().cpu(), batch["labels"].argmax(dim=1).long().cpu())
        self.log(f"train/loss", output.loss, prog_bar=False)
        self.log(f"train/f1", self.train_f1, prog_bar=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        self.zero_grad()
        output = self.forward(batch)
        output.loss = output.loss.detach()
        output.logits = output.logits.detach()
        self.log("val/loss", output.loss, prog_bar=True)
        indicators = self.logits_to_indicators(output.logits).to(
            self.device
        )
        self.val_f1(indicators.float().cpu(), batch["labels"].argmax(dim=1).long().cpu())
        self.log(f"val/loss", output.loss, prog_bar=False)
        self.log(f"val/f1", self.val_f1, prog_bar=True)

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
        indicators = torch.zeros_like(logits).long()
        indicators[logits == logits.max(dim=1)[0].unsqueeze(1)] = 1
        return indicators

    def predict(self, collection: bioc.BioCCollection) -> None:
        torch.set_grad_enabled(True)
        collator = transformers.DataCollatorWithPadding(self.tokenizer)
        it = tqdm(collection.documents, desc="Predicting")
        transformer = ExplanationGenerator.Generator(self.transformer)
        for i, doc in enumerate(it):
            for passage in doc.passages:
                for sent in passage.sentences:
                    examples = sentence_to_examples(
                        sent,
                        self.tokenizer,
                        doc_context=None,
                        pair_types=self.meta.pair_types,
                        label_to_id=None,
                        max_length=self.max_length,
                        mark_with_special_tokens=True,
                        blind_entities=False
                    )
                    if examples:
                        batch = collator([i["features"] for i in examples])
                        for k, v in batch.items():
                            batch[k] = v.to(self.device)

                        batch_explanations = []
                        self.zero_grad()
                        batch_logits = self.forward(batch).logits.detach()
                        for input_ids, attention_mask, logits in zip(batch.input_ids, batch.attention_mask, batch_logits):
                            expl = transformer.generate_LRP(input_ids=input_ids.unsqueeze(0),
                                                           attention_mask=attention_mask.unsqueeze(0),
                                                           start_layer=0).detach()
                            expl = (expl - expl.min()) / (expl.max() - expl.min())
                            assert expl.shape[1] == input_ids.shape[0]
                            batch_explanations.append(expl)
                        batch_preds = self.logits_to_indicators(batch_logits)
                        for example, input_ids, preds, logits, explanation in zip(
                                examples, batch.input_ids, batch_preds, batch_logits, batch_explanations
                        ):
                            head = example["head"]
                            tail = example["tail"]
                            for label_idx in torch.where(preds)[0]:
                                rel = bioc.BioCRelation()
                                rel.infons["prob"] = logits[label_idx.item()].item()
                                rel.infons["type"] = (
                                        self.meta.id_to_label[
                                            label_idx.item()
                                        ]
                                )
                                rel.infons["explanation"] = explanation.tolist()[0]
                                rel.infons["tokens"] = self.tokenizer.convert_ids_to_tokens(input_ids)
                                rel.add_node(bioc.BioCNode(refid=head, role="head"))
                                rel.add_node(bioc.BioCNode(refid=tail, role="tail"))
                                sent.add_relation(rel)
        return None

    def get_dataset(self, path, limit_examples=None):
        return BiocDataset(path, self.tokenizer, limit_examples=limit_examples,
                           use_doc_context=False,
                           mark_with_special_tokens=True,
                           blind_entities=False,
                           max_length=self.max_length)
