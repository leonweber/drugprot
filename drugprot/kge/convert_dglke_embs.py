import argparse
import numpy
import torch
import pickle
import shutil

from pathlib import Path

from torch import Tensor
from typing import Dict

from tqdm import tqdm

from drugprot.kge.prepare_ctd import read_parent_to_childs_mapping
from drugprot.utils import read_vocab


def convert_dglke_embeddings(emb_file: Path, data_dir: Path, output_dir: Path, extend_ctd: bool):
    print(f"Loading embeddings from {emb_file}")
    npy_data = numpy.load(str(emb_file))
    embeddings = torch.tensor(npy_data)
    print(f"Found: {embeddings.shape}")

    entity_vocab_file = data_dir / "entities.dict"
    entity_to_idx = read_vocab(entity_vocab_file)

    drug_ids = [index for id, index in entity_to_idx.items() if id.startswith("MESH")]
    gene_ids = [index for id, index in entity_to_idx.items() if id.startswith("NCBI")]

    drug_embeddings = embeddings[drug_ids]
    mean_drug_embedding = torch.mean(drug_embeddings, dim=0, keepdim=True)

    gene_embeddings = embeddings[gene_ids]
    mean_gene_embedding = torch.mean(gene_embeddings, dim=0, keepdim=True)

    embeddings = torch.cat([embeddings, mean_drug_embedding, mean_gene_embedding], dim=0)
    entity_to_idx["DRUG-UNK"] = len(entity_to_idx)
    entity_to_idx["GENE-UNK"] = len(entity_to_idx)

    if extend_ctd:
        embeddings, entity_to_idx = extend_embeddings_by_parent_child_mappings(embeddings, entity_to_idx)

    entity_emb_file = output_dir / "embeddings.pkl"
    print(f"Save embeddings to {output_dir}")
    pickle.dump(embeddings, entity_emb_file.open("wb"))

    new_entity_dict_file = output_dir / "entities.dict"
    with new_entity_dict_file.open("w") as writer:
        writer.write("\n".join([f"{id}\t{index}" for id, index in entity_to_idx.items()]))

    shutil.copyfile(data_dir / "relation.dict", output_dir / "relation.dict")


def extend_embeddings_by_parent_child_mappings(embeddings: Tensor, entity_to_idx: Dict[str, int]):
    print("Extending the embedding matrix by using the ctd chemical parent-child mapping")
    parent_to_childs = read_parent_to_childs_mapping()
    prev_size = len(entity_to_idx)


    for parent_id, childs in tqdm(parent_to_childs.items(), total=len(parent_to_childs)):
        if parent_id not in entity_to_idx:
            continue

        parent_embedding_idx = entity_to_idx[parent_id]
        parent_embedding = embeddings[parent_embedding_idx].unsqueeze(0)

        child_embeddings = torch.cat([parent_embedding]*len(childs), dim=0)
        embeddings = torch.cat([embeddings, child_embeddings], dim=0)

        for child_id in childs:
            entity_to_idx[child_id] = len(entity_to_idx)

    num_add_embeddings = len(entity_to_idx) - prev_size
    print(f"Added {num_add_embeddings} embeddings to the matrix")

    return embeddings, entity_to_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", type=Path, required=True,
                        help="Path to the entity embedding file")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to the directory containing the training files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to the output directory")
    parser.add_argument("--extend_ctd", required=False, default=True, action="store_true",
                        help="Indicates whether to extend the embedding by using the ctd chemical parent-child mapping")
    args = parser.parse_args()

    convert_dglke_embeddings(args.emb_file, args.data_dir, args.output_dir, args.extend_ctd)
