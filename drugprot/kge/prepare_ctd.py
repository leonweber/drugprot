import argparse
import pandas as pd
import shutil
import gzip

from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from drugprot.utils import download_file


def encode_dataset_for_dgl_ke(ctd_file: Path, train_size: float, output_dir: Path, random_state: float = 42):
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
            e_writer.write(entity + "\t" + str(i) + "\n")
            entity_to_id[entity] = i

    print("Encoding relations")
    relations = sorted([r for r in data["relation"].unique()])

    relation_to_id = {}
    relation_file = output_dir / "relation.dict"
    with relation_file.open("w") as r_writer:
        for i, relation in enumerate(relations):
            r_writer.write(relation + "\t" + str(i) + "\n")
            relation_to_id[relation] = i

    print(f"Building train-test split ({train_size}:{1.0-train_size})")
    if 1.0 > train_size > 0:
        train_data, test_data, _, _ = train_test_split(data, data["relation"],
                                                       train_size=train_size,
                                                       random_state=random_state)
    elif train_size == 1.0:
        print("Using complete data set for training!")
        _, test_data, _, _ = train_test_split(data, data["relation"],
                                              train_size=0.9,
                                              random_state=random_state)
        train_data = data

    else:
        raise AssertionError(f"Illegal train_size {train_size}")

    print("Encoding triplets")
    for split_file, split_data in [("train.tsv", train_data), ("valid.tsv", test_data)]:
        triplet_file = output_dir / split_file
        with triplet_file.open("w", encoding="utf8") as writer:
            for _, row in tqdm(split_data.iterrows(), total=len(data)):
                head_id = entity_to_id[row["head"]]
                tail_id = entity_to_id[row["tail"]]
                relation_id = relation_to_id[row["relation"]]
                writer.write("\t".join([str(head_id), str(relation_id), str(tail_id)]) + "\n")


def gunzip(file_path: Path, output_path: Path):
    with gzip.open(file_path,"rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", type=bool, required=False, default=True,
                        help="Indicates whether to just use human drug-chem relations")
    parser.add_argument("--output", type=Path, required=False, default=Path("data/ctd"),
                        help="Path to the output directory")
    args = parser.parse_args()

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
    print(f"Found {len(ctd_data)} entries in ctd")

    triplet_file = ctd_dir / "triplets.tsv"
    if not triplet_file.exists():
        triplet_writer = triplet_file.open("w", encoding="utf8")
        triplet_writer.write("\t".join(["head", "relation", "tail"]) + "\n")

        num_triplets = 0
        for id, row in tqdm(ctd_data.iterrows(), total=len(ctd_data)):
            if args.human and not row["OrganismID"] == 9606:
                continue

            chem_id = "MESH:" + str(row["ChemicalID"])
            target_id = "NCBI:" + str(row["GeneID"])
            interactions = row["InteractionActions"].split("|")

            for int_action in interactions:
                int_action = int_action.replace(" ", "-").strip()
                triplet_writer.write("\t".join([chem_id, int_action, target_id]) + "\n")
                num_triplets += 1

        triplet_writer.flush()
        triplet_writer.close()

        print(f"Found {num_triplets} in total")

    dglke_dir = ctd_dir / "dgl_ke"
    encode_dataset_for_dgl_ke(triplet_file, 0.9, dglke_dir, 42)

    dglke_dir = ctd_dir / "dgl_ke_full"
    encode_dataset_for_dgl_ke(triplet_file, 1.0, dglke_dir, 42)

