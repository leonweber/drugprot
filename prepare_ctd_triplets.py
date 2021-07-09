import pandas as pd
import os
import shutil
import gzip
import tempfile
import requests
import re

from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def download_file(url: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    print(cache_path)

    if cache_path.exists():
        print("File already exists in cache!")
        return cache_path

    # Download to temporary file, then copy to cache dir once finished.
    # Otherwise you get corrupt cache entries if the download gets interrupted.
    fd, temp_filename = tempfile.mkstemp()

    # GET file object
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    with open(temp_filename, "wb") as temp_file:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress.update(len(chunk))
                temp_file.write(chunk)

    progress.close()

    shutil.copyfile(temp_filename, str(cache_path))
    os.close(fd)
    os.remove(temp_filename)

    progress.close()

    return Path(cache_path)


def encode_dataset_for_dgl_ke(ctd_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(ctd_file, sep="\t")

    print("Encoding entities")
    entities = [e for e in data["head"].unique()]
    entities += [e for e in data["tail"].unique()]
    entities = sorted(set(entities))

    entity_to_id = {}
    entity_file = output_dir / "entities.dict"
    with entity_file.open("w") as e_writer:
        for i, entity in enumerate(entities):
            e_writer.write(entity + " " + str(i) + "\n")
            entity_to_id[entity] = i

    print("Encoding relations")
    relations = sorted([r for r in data["relation"].unique()])

    relation_to_id = {}
    relation_file = output_dir / "relation.dict"
    with relation_file.open("w") as r_writer:
        for i, relation in enumerate(relations):
            r_writer.write(relation + " " + str(i) + "\n")
            relation_to_id[relation] = i

    print("Encoding triplets")
    train_data, test_data, _, _ = train_test_split(data, data["relation"], train_size=0.9)
    for split_file, split_data in [("train.tsv", train_data), ("valid.tsv", test_data)]:
        triplet_file = output_dir / split_file
        with triplet_file.open("w", encoding="utf8") as writer:
            for _, row in tqdm(split_data.iterrows(), total=len(data)):
                head_id = entity_to_id[row["head"]]
                tail_id = entity_to_id[row["tail"]]
                relation_id = relation_to_id[row["relation"]]
                writer.write(" ".join([str(head_id), str(relation_id), str(tail_id)]) + "\n")

def gunzip(file_path: Path, output_path: Path):
    with gzip.open(file_path,"rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


print("Downloading ctd file")
ctd_dir = Path("data/ctd")
ctd_gz_file = ctd_dir / "CTD_chem_gene_ixns.tsv.gz"
archive_file = download_file("http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz", ctd_dir)

ctd_file = ctd_dir / archive_file.name.replace(".gz", "")
if not ctd_file.exists():
    print("Unzipping EU-ADR corpus")
    gunzip(archive_file, ctd_file)

columns = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
           "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
ctd_data = pd.read_csv(ctd_file, sep="\t", comment="#", names=columns)
print(len(ctd_data))

triplet_file = ctd_dir / "triplets.tsv"
if not triplet_file.exists():
    triplet_writer = triplet_file.open("w", encoding="utf8")
    triplet_writer.write("\t".join(["head", "relation", "tail"]) + "\n")

    num_triplets = 0
    for id, row in tqdm(ctd_data.iterrows(), total=len(ctd_data)):
        if not row["OrganismID"] == 9606:
            continue

        chem_id = "MESH:" + str(row["ChemicalID"])
        target_id = "NCBI:" + str(row["GeneID"])
        interactions = row["InteractionActions"].split("|")

        for int_action in interactions:
            int_action = int_action.replace(" ", "-").strip()
            triplet_writer.write("\t".join([chem_id, target_id, int_action]) + "\n")
            num_triplets += 1

    triplet_writer.flush()
    triplet_writer.close()

    print(f"Found {num_triplets} in total")

dglke_dir = ctd_dir / "dgl_ke"
encode_dataset_for_dgl_ke(triplet_file, dglke_dir)

