import argparse
import math

from collections import defaultdict
from pathlib import Path


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def read_reverse_vocabulary(vocab_file: Path):
    print(f"Reading vocabulary from {vocab_file}")

    id_to_concept = {}
    with vocab_file.open("r", encoding="utf8") as reader:
        for line in reader.readlines():
            key, value = line.strip().split("\t")
            id_to_concept[int(value)] = key

    print(f"Found {len(id_to_concept)} vocab entries")
    return id_to_concept


def aggregate_results(result_file: Path, entity_dict_file: Path, relation_dict: Path,
                      output_file: Path, ct_threshold: float):

    id_to_entity = read_reverse_vocabulary(entity_dict_file)
    id_to_relation = read_reverse_vocabulary(relation_dict)

    pair_to_relations = {}

    print("Aggregating prediction results")
    with result_file.open("r", encoding="utf8") as reader:
        for i, line in enumerate(reader.readlines()):
            if i == 0:
                continue

            head, rel, tail, score = line.strip().split("\t")

            head = id_to_entity[int(head)]
            tail = id_to_entity[int(tail)]
            pair_id = head + "#" + tail

            if pair_id not in pair_to_relations:
                pair_to_relations[pair_id] = []

            confidence = sigmoid(float(score))
            if confidence <= ct_threshold:
                continue

            rel = id_to_relation[int(rel)]
            pair_to_relations[pair_id] += [(rel, str(confidence))]

    with output_file.open("w", encoding="utf8") as writer:
        writer.write("\t".join(["head", "tail", "in_relation", "relations"]) + "\n")
        for pair_id, relations in pair_to_relations.items():
            head, tail = pair_id.split("#")
            in_relation = "1" if len(relations) > 0 else "0"

            rel_str = ""
            if len(relations) > 0:
                rel_str = "|".join([f"({rel}, {conf})" for rel, conf in relations])

            writer.write("\t".join([head, tail, in_relation, rel_str]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True,
                        help="Path to the result file produced by dglke_predict")
    parser.add_argument("--entity_dict", type=Path, required=True,
                        help="Path to entity dictionary file")
    parser.add_argument("--relation_dict", type=Path, required=True,
                        help="Path to relation dictionary file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to the output file")
    parser.add_argument("--threshold", type=float, required=False, default=0.5,
                        help="Confidence threshold for positive predictions")
    args = parser.parse_args()

    aggregate_results(args.result, args.entity_dict, args.relation_dict, args.output, args.threshold)
