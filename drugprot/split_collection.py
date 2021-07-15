from argparse import ArgumentParser
from pathlib import Path

import bioc
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train_output", type=Path, required=True)
    parser.add_argument("--test_output", type=Path, required=True)
    parser.add_argument("--train_size", type=float, default=0.9)
    args = parser.parse_args()

    with args.input.open() as f:
        collection = bioc.load(f)

    train_docs, test_docs = train_test_split(collection.documents,
                                             train_size=args.train_size)
    train_collection = bioc.BioCCollection()
    train_collection.documents = train_docs

    test_collection = bioc.BioCCollection()
    test_collection.documents = test_docs

    with args.train_output.open("w") as f:
        bioc.dump(train_collection, f)

    with args.test_output.open("w") as f:
        bioc.dump(test_collection, f)

