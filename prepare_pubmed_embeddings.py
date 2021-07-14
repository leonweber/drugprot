import random
import torch
import numpy as np
import pickle

from pathlib import Path
from typing import Dict, List
from gensim.models import KeyedVectors
from torch import Tensor
from tqdm import tqdm

from drugprot.utils import download_file


def read_vocabulary(vocab_file: Path) -> Dict[str, int]:
    vocab = {}
    with vocab_file.open("r", encoding="utf8") as reader:
        for i, line in enumerate(reader.readlines()):
            vocab[line.strip()] = i

    return vocab


def read_drug_mesh_mapping(mapping_file: Path) -> Dict[str, str]:
    drug_to_mesh = {}
    with mapping_file.open("r", encoding="utf8") as reader:
        for i, line in enumerate(reader.readlines()):
            if i == 0:
                continue

            mesh_id, db_id = line.strip().split("\t")
            mesh_id = mesh_id.strip()
            db_id = db_id.strip()

            if db_id in drug_to_mesh:
                print(f"Found multiple mappings for {db_id}")

            drug_to_mesh[db_id] = mesh_id

    return drug_to_mesh


def read_embeddings(emb_file: Path) -> torch.Tensor:
    print(f"Reading embeddings from {emb_file}")
    doc2vec_model = KeyedVectors.load(str(emb_file))
    print(f"Found {doc2vec_model.count} embeddings")

    return torch.tensor(doc2vec_model.vectors_docs)


def map_drug_embeddings(drug_emb_file: Path, drug_to_id: Dict[str, int], drug_to_mesh: Dict[str, str]):
    drug_embeddings = read_embeddings(drug_emb_file)

    print("Mapping drug embeddings to mesh")
    mappable_drugs = [drug for drug in drug_to_id.keys() if drug in drug_to_mesh]
    num_mappable_drugs = len(mappable_drugs)
    print(f"Found a mapping for {num_mappable_drugs} (out of {drug_embeddings.shape[0]}) drugs")

    new_embeddings = np.zeros([num_mappable_drugs, drug_embeddings.shape[1]])
    new_mapping = {}
    for i, drug_id in tqdm(enumerate(mappable_drugs), total=num_mappable_drugs):
        drug_index = drug_to_id[drug_id]
        new_embeddings[i] = drug_embeddings[drug_index]
        new_mapping[drug_id] = i

    return new_mapping, torch.tensor(new_embeddings)


def combine_embeddings(dictionaries: List[Dict[str, int]], embeddings: List[Tensor]):
    all_embeddings = torch.cat(embeddings, dim=0)

    combined_dict = {key: value for key, value in dictionaries[0].items()}
    for dictionary in dictionaries[1:]:
        global_id = len(combined_dict)
        for key, value in dictionary.items():
            combined_dict[key] = value + global_id

    return combined_dict, all_embeddings


def perform_sanity_check(entity_to_id: Dict[str, int], entity_embeddings: Tensor,
                         combined_to_id: Dict[str, int], combined_embeddings: Tensor,
                         samples: int):

    entity_sample = random.sample(list(entity_to_id.keys()), samples)

    for entity_id in tqdm(entity_sample, total=len(entity_sample)):
        orig_emb = entity_embeddings[entity_to_id[entity_id]]
        comb_emb = combined_embeddings[combined_to_id[entity_id]]

        for i in range(len(orig_emb)):
            assert orig_emb[i] == comb_emb[i]


def save(dictionary: Dict[str, int], embeddings: Tensor, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    dict_file = output_dir / "entities.dict"
    with dict_file.open("w", encoding="utf8") as writer:
        for key, value in dictionary.items():
            writer.write("\t".join([key, str(value)]) + "\n")

    emb_file = output_dir / "entity_embeddings.pkl"
    pickle.dump(embeddings, emb_file.open("wb"))


pubtator_dir = Path("data/pubtator")
pubtator_dir.mkdir(parents=True, exist_ok=True)

print("Download gene embeddings and vocabulary")
gene_emb_url = "https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v2000.bin"
download_file(gene_emb_url, pubtator_dir)
gene_vocab_url = "https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/gene/gene-v0500.vocab"
download_file(gene_vocab_url, pubtator_dir)

print("Download drug embeddings and vocabulary")
drug_emb_url = "https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v2000.bin"
download_file(drug_emb_url, pubtator_dir)
drug_vocab_url = "https://www2.informatik.hu-berlin.de/~saengema/bio-entity-embeddings/v1/drug/drug-v0500.vocab"
download_file(drug_vocab_url, pubtator_dir)

gene_vocab_file = pubtator_dir / "gene-v0500.vocab"
gene_to_id = read_vocabulary(gene_vocab_file)

gene_emb_file = pubtator_dir / "gene-v2000.bin"
gene_embeddings = read_embeddings(gene_emb_file)

drug_vocab_file = pubtator_dir / "drug-v0500.vocab"
drug_to_id = read_vocabulary(drug_vocab_file)

drug_to_mesh = read_drug_mesh_mapping(Path("data/drug_mapping.tsv"))
drug_emb_file = pubtator_dir / "drug-v2000.bin"

drug_to_id, drug_embeddings = map_drug_embeddings(drug_emb_file, drug_to_id, drug_to_mesh)

entity_to_id, entity_embeddings = combine_embeddings([drug_to_id, gene_to_id], [drug_embeddings, gene_embeddings])

print("Perform drug sanity checks")
perform_sanity_check(drug_to_id, drug_embeddings, entity_to_id, entity_embeddings, len(drug_to_id))

print("Perform gene sanity checks")
perform_sanity_check(gene_to_id, gene_embeddings, entity_to_id, entity_embeddings, int(len(gene_to_id) / 22.5))

max_entity_id = max(entity_to_id.values())
assert entity_embeddings[max_entity_id] is not None

output_dir = Path("data/embeddings/pubtator_embeddings")
save(entity_to_id, entity_embeddings, output_dir)

