from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import bioc
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print("predictions: ", args.input)
    doc_id_to_docs = defaultdict(list)
    for input_file in args.input:
        with input_file.open() as f:
            docs = bioc.load(f).documents
            for doc in docs:
                doc_id_to_docs[doc.id].append(doc)


    with args.output.open("w") as f:
        for doc_id, docs in doc_id_to_docs.items():
            for i_sentence, sentence in enumerate(docs[0].passages[0].sentences):
                rel_to_probs = defaultdict(list)
                for doc in docs:
                    for rel in doc.passages[0].sentences[i_sentence].relations:
                        if "prob" in rel.infons:
                            head = rel.get_node("head").refid
                            tail = rel.get_node("tail").refid
                            rel_type = rel.infons["type"]
                            rel_to_probs[(head, rel_type, tail)].append(float(rel.infons["prob"]))
                for (head, rel, tail), probs in rel_to_probs.items():
                    prob = np.mean(probs)
                    if prob > 0.5 and (rel.startswith("drugprot/") or "/" not in rel):
                        rel = rel.replace("drugprot/", "")
                        if rel.lower() != "none":
                            f.write(f"{doc.id}\t{rel}\targ1:{head}\targ2:{tail}\n")
