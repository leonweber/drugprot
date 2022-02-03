import gzip
from collections import defaultdict

from flair.file_utils import cached_path
import pandas as pd

def get_uniprot_to_entrez():
    numeric_to_uniprot = {}
    numeric_to_entrez = defaultdict(set)
    with gzip.open(cached_path("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz", "data")) as f:
        for line in f:
            fields = line.strip().decode().split("\t")
            if not len(fields) == 3:
                continue
            head, rel, tail = fields
            if rel == "UniProtKB-ID":
                numeric_to_uniprot[head] = tail
            elif rel == "GeneID":
                numeric_to_entrez[head].add(tail)

    uniprot_to_entrez = defaultdict(set)
    for numeric, uniprot in numeric_to_uniprot.items():
        uniprot_to_entrez[uniprot].update(numeric_to_entrez[numeric])

    return uniprot_to_entrez


def get_ttd_to_geneid():
    ttd_to_geneid = defaultdict(set)
    uniprot_to_entrez = get_uniprot_to_entrez()
    with open(cached_path("http://db.idrblab.net/ttd/sites/default/files/ttd_database/P1-01-TTD_target_download.txt", "data")) as f:
        for line in f:
            fields = line.strip().split("\t")
            if not len(fields) == 3:
                continue
            head, rel, tail = fields
            if rel == "UNIPROID":
                ttd_to_geneid[head].update(uniprot_to_entrez[tail])

    return ttd_to_geneid


def get_pubchem_to_mesh():
    pubchem_to_mesh = defaultdict(set)
    total = 0
    unmappable = 0

    name_to_mesh = {}
    df_ctd = pd.read_csv("http://ctdbase.org/reports/CTD_chemicals.tsv.gz",
                     compression="gzip", sep="\t", comment="#", header=None)
    mesh_names = df_ctd.iloc[:, 0]
    mesh_ids = df_ctd.iloc[:, 1]
    for name, mesh in zip(mesh_names, mesh_ids):
        assert name not in name_to_mesh
        name_to_mesh[name] = mesh

    with open(cached_path("https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-MeSH", cache_dir="data")) as f:
        for line in f:
            fields = line.strip().split("\t")
            for name in fields[1:]:
                total += 1
                if name in name_to_mesh:
                    pubchem_to_mesh[fields[0]].add(name_to_mesh[name])
                else:
                    unmappable += 1

    print(f"{unmappable} out of {total} PubChem names could not be mapped to MeSH")

    return pubchem_to_mesh

def get_ttd_to_mesh():
    ttd_to_mesh = defaultdict(set)

    pubchem_to_mesh = get_pubchem_to_mesh()

    all_ttd_ids = set()

    num_ttd_with_pubchem = 0

    with open(cached_path("http://db.idrblab.net/ttd/sites/default/files/ttd_database/P1-03-TTD_crossmatching.txt", "data")) as f:
        for line in f:
            fields = line.strip().split("\t")
            if not len(fields) == 3:
                continue
            head, rel, tail = fields
            all_ttd_ids.add(head)
            if rel == "PUBCHCID":
                ttd_to_mesh[head].update(pubchem_to_mesh[tail.replace("CAS ", "")])
                num_ttd_with_pubchem += 1

    num_ttd_with_mesh = len([i for i in ttd_to_mesh.values() if i])
    print(f"{num_ttd_with_mesh} of {len(all_ttd_ids)} TTD ids have a MESH id")

    return ttd_to_mesh


if __name__ == '__main__':
    triples = set()
    n_mappable = 0
    df = pd.read_excel("http://db.idrblab.net/ttd/sites/default/files/ttd_database/P1-07-Drug-TargetMapping.xlsx")
    ttd_to_geneid = get_ttd_to_geneid()
    ttd_to_mesh = get_ttd_to_mesh()

    for chem, rel, prot in zip(df["DrugID"], df["MOA"], df["TargetID"]):
        mappable = False
        for chem_id in ttd_to_mesh[chem]:
            for gene_id in ttd_to_geneid[prot]:
                triples.add((chem_id, rel, gene_id))
                mappable = True

        if mappable:
            n_mappable += 1
    print(f"Could map {n_mappable} of {len(df)} TTD triples to MESH / gene id namespace")

    with open("data/ttd.tsv", "w") as f:
        for triple in triples:
            f.write("\t".join(triple) + "\n")
