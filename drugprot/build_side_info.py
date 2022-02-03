import gzip
import shutil
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import io
import urllib
from urllib import request

import pandas as pd
from lxml import etree
from tqdm import tqdm

from transformers import cached_path

if __name__ == "__main__":

    mesh_to_description = {}
    mesh_to_parents = {}
    mesh_parent_to_children = defaultdict(set)
    geneid_to_description = {}
    mesh_geneid_to_interaction = {}

    with gzip.open(cached_path("http://ctdbase.org/reports/CTD_chemicals.csv.gz")) as f:
        df = pd.read_csv(f, comment="#", sep=",", header=None)
        df.columns = ["ChemicalName", "ChemicalID", "CasRN", "Definition", "ParentIDs", "TreeNumbers", "ParentTreeNumbers", "Synonyms"]
        df = df.fillna("")
        for _, row in df.iterrows():![](../../../Documents/Student projects/2021 Xing Masterarbeit/Masterarbeit_Xing_David_Wang.png)
            mesh_to_description[row["ChemicalID"]] = row["Definition"]
            mesh_to_parents[row["ChemicalID"]] = row["ParentIDs"].split("|")
            for parent_mesh in row["ParentIDs"].split("|"):
                mesh_parent_to_children[parent_mesh].add(row["ChemicalID"])

    print(f"{len([i for i in mesh_to_description.values() if i])} / {len(mesh_to_description)} MESH-IDs have a description before parent expansion")

    for mesh, description in mesh_to_description.items():
        if not description:
            for parent in mesh_to_parents[mesh]:
                if not parent:
                    continue
                parent_description = mesh_to_description[parent]
                if parent_description:
                    mesh_to_description[mesh] = parent_description
                    break
    print(f"{len([i for i in mesh_to_description.values() if i])} / {len(mesh_to_description)} MESH-IDs have a description after parent expansion")

    with open("data/side_information/ctd_entities.tsv", "w") as f:
        for mesh, description in mesh_to_description.items():
            mesh = mesh.replace("MESH:", "")
            f.write(f"{mesh}\t{description}\n")

    with gzip.open(cached_path("http://ctdbase.org/reports/CTD_chem_gene_ixns.csv.gz")) as f:
        df = pd.read_csv(f, comment="#", sep=",", header=None)
        df.columns = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms", "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
        for _, row in tqdm(df.iterrows()):
            mesh_geneid_to_interaction[(row["ChemicalID"], row["GeneID"])] = row["InteractionActions"].replace("^", " ").replace("|", ", ")

        for (mesh_id, gene_id), interactions in tqdm(list(mesh_geneid_to_interaction.items())):
            for child in mesh_parent_to_children[mesh_id]:
                if (child, gene_id) not in mesh_geneid_to_interaction:
                    mesh_geneid_to_interaction[(child, gene_id)] = interactions

    with open("data/side_information/ctd_relations.tsv", "w") as f:
        for (mesh, gene_id), interactions in mesh_geneid_to_interaction.items():
            f.write(f"{mesh}\t{gene_id}\t{interactions}\n")

    # with closing(request.urlopen("https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz")) as r:
    #     nsmap = {"up": "http://uniprot.org/uniprot"}
    #     with gzip.open(r) as f:
    #         tree = etree.parse(f)
    #         entries = tree.findall("up:entry", namespaces=nsmap)
    #         for entry in entries:
    #             gene_id = entry.xpath("up:dbReference[@type='GeneID']/@id", namespaces=nsmap)
    #             function = entry.xpath("up:comment[@type='function']/up:text/text()", namespaces=nsmap)
    #             if gene_id and function:
    #                 geneid_to_description[gene_id[0]] = function[0]
    # with open("data/side_information/uniprot_entities.tsv", "w") as f:
    #     for geneid, description in geneid_to_description.items():
    #         f.write(f"{geneid}\t{description}\n")
    #
    #
    #
