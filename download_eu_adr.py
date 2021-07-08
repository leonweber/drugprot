import os
import re
import requests
import shutil
import tarfile
import tempfile
import time

from lxml import etree
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List


def download_file(url: str, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = re.sub(r".+/", "", url)
    # get cache path to put the file
    cache_path = cache_dir / filename
    print(cache_path)

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


def get_pubmed_abstract(pubmed_id: str, output_dir: Path):
    output_file = output_dir / f"{pubmed_id}.xml"
    if not output_file.exists():
        response = requests.get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
                                f"db=pubmed&id={pubmed_id}&retmode=XML&rettype=abstract")

        if response.status_code == 200:
            output_file.write_text(str(response.content.decode("utf8")), encoding="utf8")
        else:
            print(f"Can't fetch abstract for pubmed id {pubmed_id} (reponse code: {response.status_code})")
            return

    root = etree.parse(output_file.open("r", encoding="utf8"))
    title_elements = root.xpath("//ArticleTitle")
    title = " ".join([e.text for e in title_elements])
    title = title.replace("\t", " ")
    title.strip()

    abstract_elements = root.xpath("//AbstractText")
    abstract = ""
    for element in abstract_elements:
        if "Label" in element.attrib:
            abstract += f" {element.attrib['Label']}:"
            abstract = abstract.strip()
        abstract += " " + element.text

    abstract = abstract.replace("\t", " ")
    abstract = abstract.strip()

    abstract_file = output_dir / f"{pubmed_id}.txt"
    abstract_file.write_text(title + "\t" + abstract)


def load_texts(text_dir: Path):
    pmid_to_text = {}
    for file in text_dir.iterdir():
        if not file.name.endswith(".txt"):
            continue

        pmid = file.name.replace(".txt", "")
        title, abstract = file.read_text(encoding="utf8").strip().split("\t")
        text = title + " " + abstract
        pmid_to_text[pmid] = [text, title, abstract]

    return pmid_to_text


def load_entity_annotations(data_dir: Path):
    pmid_to_entities = defaultdict(list)
    for file in data_dir.iterdir():
        if not file.name.endswith(".ann"):
            continue

        pmid = file.name.replace(".ann", "")
        has_relation = False
        entities = []

        if pmid == "18234154":
            print(pmid)

        with file.open("r") as reader:
            for line in reader.readlines():
                cols = line.strip().split("\t")

                if cols[0] == "Target-Drug" and cols[1] == "True" and cols[2] == "relation":
                    has_relation = True
                    continue

                if not (cols[1] == "True" and cols[2] == "concept"):
                    continue

                mention_text = cols[3].strip()
                start, end = int(cols[4]), int(cols[5])
                entity_id = int(cols[8])
                entity_type = cols[9]
                if entity_type == "Chemicals & Drugs":
                    entity_type = "CHEMICAL"
                elif entity_type == "Genes & Molecular Sequences":
                    entity_type = "GENE-Y"
                else:
                    continue
                entities.append((start, end, mention_text, entity_type, entity_id))

            if pmid == "18234154":
                print(pmid)

        if has_relation:
            pmid_to_entities[pmid] = entities

    return pmid_to_entities


def load_relations(data_dir: Path):
    pmid_to_relations = defaultdict(list)
    num_negative = 0
    num_positive = 0

    for file in data_dir.iterdir():
        if not file.name.endswith(".ann"):
            continue

        pmid = file.name.replace(".ann", "")
        relations = []

        with file.open("r") as reader:
            for line in reader.readlines():
                cols = line.strip().split("\t")

                if not (cols[0] == "Target-Drug" and cols[1] == "True" and cols[2] == "relation"):
                    continue

                source_id, target_id = cols[3], cols[4]
                assessment = cols[10]
                if assessment == "PA" or assessment == "SA":
                    relations += [(source_id, target_id)]
                    num_positive += 1
                elif assessment == "NA":
                    print(f"Negative: {pmid}")
                    num_negative += 1
                else:
                    raise Exception()

        if len(relations) > 0:
            pmid_to_relations[pmid] = relations

    print(f"Positive: {num_positive}")
    print(f"Negative: {num_negative}")
    return pmid_to_relations


def fix_annotations(pmid_to_text: Dict[str, List[str]], pmid_to_entities: Dict[str, List]):
    num_not_mappable = 0
    pmid_to_entities_clean = {}
    for pmid, entities in pmid_to_entities.items():
        text = pmid_to_text[pmid][0]
        cleaned_entities = []

        for (start, end, mention, type, id) in entities:
            if len(mention.strip()) == 0:
                continue

            ann_text = text[start:end]
            if ann_text == mention:
                cleaned_entities += [(start, end, mention, type, id)]
                continue

            ann_text = text[start:start + len(mention)]
            if ann_text == mention:
                cleaned_entities += [(start, start + len(mention), mention, type, id)]
                continue

            ann_text = text[end - len(mention):end]
            if ann_text == mention:
                cleaned_entities += [(end - len(mention), end, mention, type, id)]
                continue

            num_not_mappable += 1
            print(f"Not mappable: {text[start:end]}")
            cleaned_entities += [(start, end, mention, type, id)]

        if len(entities) > 0:
            pmid_to_entities_clean[pmid] = cleaned_entities

    print(f"Not mappable: {num_not_mappable}")
    return pmid_to_entities_clean


def generate_pseudo_bart(pmid_to_text: Dict[str, List[str]], pmid_to_entities: Dict[str, List],
                         pmid_to_relations: Dict[str, List], output_dir: Path):

    text_writer = (output_dir / "euadr_abstracts.tsv").open("w", encoding="utf8")
    entity_writer = (output_dir / "euadr_entities.tsv").open("w", encoding="utf8")
    relation_writer = (output_dir / "euadr_relations.tsv").open("w", encoding="utf8")

    for pmid, relations in pmid_to_relations.items():
        _, title, abstract = pmid_to_text[pmid]
        entities = pmid_to_entities[pmid]

        if pmid == "18234154":
            print(pmid)

        text_writer.write("\t".join([pmid, title, abstract]) + "\n")

        for (start, end, mention, type, id) in entities:
            entity_writer.write("\t".join([pmid, f"T{id}", type, str(start), str(end), mention]) + "\n")

        for (source, target) in relations:
            relation_writer.write("\t".join([pmid, "DRUG-TARGET", f"Arg1:T{source}", f"Arg2:T{target}"]) + "\n")

    for writer in [text_writer, entity_writer, relation_writer]:
        writer.flush()
        writer.close()

print("Downloading EU-ADR corpus")
eu_adr_dir = Path("data/eu_adr")
eu_adr_dir.mkdir(parents=True, exist_ok=True)
eu_adr_file = download_file("https://biosemantics.erasmusmc.nl/downloads/euadr.tgz", eu_adr_dir)

print("Unzipping EU-ADR corpus")
tar = tarfile.open(eu_adr_file, "r:gz")
tar.extractall(eu_adr_dir)
tar.close()

print("Cleaning downloaded files")
data_dir = eu_adr_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

extracted_dir = eu_adr_dir / "euadr_corpus"
for file in extracted_dir.iterdir():
    if not file.name.startswith("._"):
        shutil.move(file, data_dir / file.name.replace(".txt", ".ann"))
    else:
        file.unlink()

extracted_dir.rmdir()

print("Fetching abstracts")
text_dir = eu_adr_dir / "data"
text_dir.mkdir(parents=True, exist_ok=True)
for file in tqdm(data_dir.iterdir(), total=300):
    if not file.name.endswith(".ann"):
        continue

    pubmed_id = file.name.replace(".ann", "")
    abstract_file = text_dir / (pubmed_id + ".txt")
    if not abstract_file.exists():
        get_pubmed_abstract(pubmed_id, text_dir)
        time.sleep(0.02)

print("Cleaning annotation")
pmid_to_entities = load_entity_annotations(data_dir)
pmid_to_text = load_texts(text_dir)

print("Fixing annotation")
pmid_to_entities = fix_annotations(pmid_to_text, pmid_to_entities)

print("Loading relations")
pmid_to_relations = load_relations(data_dir)

print(f"Writing pseudo brat output ({len(pmid_to_relations.keys())} documents)")
brat_dir = eu_adr_dir / "brat"
brat_dir.mkdir(parents=True, exist_ok=True)
generate_pseudo_bart(pmid_to_text, pmid_to_entities, pmid_to_relations, brat_dir)
