#!/usr/bin/env python
# coding: utf-8


from transformers import AutoTokenizer
import transformers
from drugprot.models.explainable_bert import ExplainableBert
from drugprot import utils
import numpy as np
from drugprot.explain.BERT import ExplanationGenerator
from tqdm import tqdm
import torch


# In[3]:


tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_fast=True)
tokenizer.add_tokens(
    ['<e1>', '</e1>', '<e2>', '</e2>'], special_tokens=True
)
collator = transformers.DataCollatorWithPadding(tokenizer)


# In[4]:


dataset_to_meta = {"drugprot": utils.get_dataset_metadata("data/drugprot/dev.bioc.xml")}
model = ExplainableBert.load_from_checkpoint("saved_runs/explainable_bert/checkpoints/epoch=02.ckpt",
                                            transformer="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                                            lr=0.01, finetune_lr=0.01, dataset_to_meta=dataset_to_meta,
                                            max_length=128, optimized_metric="").cuda()
explainer = ExplanationGenerator.Generator(model.transformer)


# In[56]:


examples = []
processed_sentences = set()
with open("/vol/fob-vol1/mi15/weberple/projects/drugprot/data/ctd/small.tsv") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        _, head_cuid, _, tail_cuid, labels, text, _ = line.split("\t")
        features = tokenizer.encode_plus(text, max_length=128, truncation=True)
        text = tokenizer.decode(features.input_ids)
        if text not in processed_sentences and '</e1>' in text and '</e2>' in text:
            processed_sentences.add(text)
            example = {
                "features": features,
                "head_cuid": head_cuid,
                "tail_cuid": tail_cuid,
                "labels": labels
            }
            examples.append(example)


# In[54]:


def get_explanation_tokens(explanation, tokens, topk):
    explanation_tokens = []
    explanation = np.array(explanation)
    for i in np.argsort(explanation)[-topk:][::-1]:
        token = tokens[i]
        if token != "[PAD]":
            explanation_tokens.append(token)

    return explanation_tokens


# In[64]:


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

with open("/vol/fob-vol1/mi15/weberple/projects/drugprot/data/ctd/small_positive_explained.tsv", "w") as f:
    meta = dataset_to_meta["drugprot"]
    for batch_examples in tqdm(list(chunks(examples, 16))):
        batch = collator([i["features"] for i in batch_examples])
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        batch_explanations = []
        model.zero_grad()
        batch_logits = model.forward(batch).logits.detach()

        for input_ids, attention_mask, logits, example in zip(batch.input_ids, batch.attention_mask, batch_logits, batch_examples):
            if logits.argmax() == 0:
                continue

            expl = explainer.generate_LRP(input_ids=input_ids.unsqueeze(0),
                                           attention_mask=attention_mask.unsqueeze(0),
                                           start_layer=0).detach().cpu().numpy()[0]
            expl = (expl - expl.min()) / (expl.max() - expl.min())
            explanation_tokens = get_explanation_tokens(expl, tokenizer.convert_ids_to_tokens(input_ids), topk=10)
            text = tokenizer.decode(input_ids).replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "")
            labels = example["labels"]
            head_cuid = example["head_cuid"]
            tail_cuid = example["tail_cuid"]
            f.write(f"{text}\t{meta.id_to_label[logits.argmax().item()]}\t{torch.softmax(logits, dim=0).max().item()}\t{' '.join(explanation_tokens)}\t{head_cuid}\t{tail_cuid}\t{labels}\n")

