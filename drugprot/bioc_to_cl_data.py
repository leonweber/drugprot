import argparse
import bioc

from collections import defaultdict
from pathlib import Path


def prepare_data(input_bioc: Path, output_file: Path):
    with input_bioc.open("r", encoding="utf8") as fp:
        collection = bioc.load(fp)

    num_drugs = 0
    num_genes = 0
    num_drugs_wo_id = 0
    num_genes_wo_id = 0

    all_relations = {}
    for document in collection.documents:
        for passage in document.passages:
            for sentence in passage.sentences:
                drugs = {}
                genes = {}

                for annotation in sentence.annotations:
                    entity = [annotation.id, annotation.text, annotation.infons["identifier"]]
                    type = annotation.infons["type"]
                    if type == "CHEMICAL":
                        if entity[2].startswith("D") or entity[2].startswith("C"):
                            drugs[entity[0]] = entity
                            num_drugs += 1
                        else:
                            num_drugs_wo_id += 1

                    elif type.startswith("GENE"):
                        first_id = entity[2].split("|")[0]
                        if first_id.isnumeric():
                            entity[2] = first_id #FIXME:!
                            genes[entity[0]] = entity
                            num_genes += 1
                        else:
                            num_genes_wo_id += 1
                    else:
                        print(f"Found unsupported entity type: {type}")

                relations = defaultdict(list)
                for relation in sentence.relations:
                    relation_type = relation.infons["type"]
                    head_id = [n.refid for n in relation.nodes if n.role == "head"][0]
                    tail_id = [n.refid for n in relation.nodes if n.role == "tail"][0]
                    relations[head_id + "#" + tail_id] += [relation_type]

                for drug_id, drug_entity in drugs.items():
                    for gene_id, gene_entity in genes.items():
                        annotation_pair_id = drug_id + "#" + gene_id
                        sent_relations = relations[annotation_pair_id]

                        entity_pair_id = drug_entity[2] + "#" + gene_entity[2]
                        if entity_pair_id not in all_relations:
                            all_relations[entity_pair_id] = set()
                        for rel in sent_relations:
                            all_relations[entity_pair_id].add(rel)

    print(f"Found {num_drugs} valid drugs ({num_drugs_wo_id} without id)")
    print(f"Found {num_genes} valid genes ({num_genes_wo_id} without id)")

    with output_file.open("w", encoding="utf8") as writer:
        writer.write("\t".join(["head", "tail", "relations"]) + "\n")

        for pair_id, relations in all_relations.items():
            head, tail = pair_id.split("#")
            rel_str = "|".join(relations)
            writer.write("\t".join(["MESH:" + head, "NCBI:" + tail, rel_str]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_bioc", type=Path, required=True,
                        help="Path to the input bioc file")
    parser.add_argument("--output_file", type=Path, required=True,
                        help="Path to the output file")
    args = parser.parse_args()

    prepare_data(args.input_bioc, args.output_file)
