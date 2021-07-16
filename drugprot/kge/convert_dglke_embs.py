import argparse
import numpy
import torch
import pickle
import shutil

from typing import Dict
from pathlib import Path
from torch import Tensor

from drugprot.utils import read_vocab


def convert_dglke_embeddings(emb_file: Path, data_dir: Path, output_dir: Path):
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

    entity_emb_file = output_dir / "embeddings.pkl"
    print(f"Save embeddings to {output_dir}")
    pickle.dump(embeddings, entity_emb_file.open("wb"))

    new_entity_dict_file = output_dir / "entities.dict"
    with new_entity_dict_file.open("w") as writer:
        writer.write("\n".join([f"{id}\t{index}" for id, index in entity_to_idx.items()]))

    shutil.copyfile(data_dir / "relation.dict", output_dir / "relation.dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", type=Path, required=True,
                        help="Path to the entity embedding file")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to the directory containing the training files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to the output directory")
    args = parser.parse_args()

    convert_dglke_embeddings(args.emb_file, args.data_dir, args.output_dir)
