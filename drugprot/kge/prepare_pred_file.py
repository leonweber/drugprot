import argparse

from pathlib import Path
from tqdm import tqdm

from drugprot.utils import read_vocab

def prepare_data(input_file: Path, entity_dict_file: Path, relation_dict_file: Path, output_dir: Path):
    entity_to_id = read_vocab(entity_dict_file)
    relations = [str(rel_id) for rel_id in read_vocab(relation_dict_file).values()]
    num_relations = len(relations)

    print(f"Reading pairs from {input_file}")
    num_not_mappable = 0
    pairs = []
    with input_file.open("r", encoding="utf8") as reader:
        for line in reader.readlines():
            head, tail = line.strip().split(" ")
            if head not in entity_to_id or tail not in entity_to_id:
                num_not_mappable += 1
                continue
            pairs += [(head, tail)]
    print(f"Found {len(pairs)} valid pairs ({num_not_mappable} pairs couldn't be mapped)")

    head_file = output_dir / "head.list"
    head_writer = head_file.open("w", encoding="utf8")
    relation_file = output_dir / "rel.list"
    relation_writer = relation_file.open("w", encoding="utf8")
    tail_file = output_dir / "tail.list"
    tail_writer = tail_file.open("w", encoding="utf8")

    print("Encoding pairs")
    for head, tail in tqdm(pairs, total=len(pairs)):
        head_id = str(entity_to_id[head])
        head_writer.write("\n".join([head_id] * num_relations) + "\n")

        relation_writer.write("\n".join(relations) + "\n")

        tail_id = str(entity_to_id[tail])
        tail_writer.write("\n".join([tail_id] * num_relations) + "\n")

    for writer in [head_writer, relation_writer, tail_writer]:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pairs", type=Path, required=True,
                        help="Path to the input file containing the IDs of the to-be-classified pairs")
    parser.add_argument("--entity_dict", type=Path, required=True,
                        help="Path to entity dictionary file")
    parser.add_argument("--relation_dict", type=Path, required=True,
                        help="Path to relation dictionary file")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to directory where the result files should be saved")

    args = parser.parse_args()

    prepare_data(args.input_pairs, args.entity_dict, args.relation_dict, args.output_dir)
