from argparse import ArgumentParser
from pathlib import Path

import bioc

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open() as f:
        collection = bioc.load(f)

    with args.output.open("w") as f:
        for doc in collection.documents:
            for sentence in doc.passages[0].sentences:
                for rel in sentence.relations:
                    if "prob" in rel.infons:
                        head = rel.get_node("head").refid
                        tail = rel.get_node("tail").refid
                        rel = rel.infons["type"]
                        if rel.startswith("drugprot/"):
                            rel = rel.replace("drugprot/", "")
                            if rel.lower() != "none":
                                f.write(f"{doc.id}\t{rel}\tArg1:{head}\tArg2:{tail}\n")

