from argparse import ArgumentParser
from pathlib import Path
from typing import List

import bioc
from lxml import etree



def get_bioc_from_ddi_file(file: Path) -> bioc.BioCDocument:
    bioc_doc = bioc.BioCDocument()
    bioc_passage = bioc.BioCPassage()
    bioc_passage.offset = 0
    bioc_doc.add_passage(bioc_passage)

    root = etree.parse(str(file.absolute()))
    documents = root.xpath("//document")
    assert len(documents) == 1

    ddi_id_to_bioc_id = {}

    bioc_doc.id = documents[0].attrib["id"]

    sentence_texts = []
    for idx_sentence, sentence in enumerate(documents[0].xpath(".//sentence")):
        bioc_sentence = bioc.BioCSentence()
        bioc_sentence.offset = len(" ".join(sentence_texts))
        if bioc_sentence.offset > 0:
            bioc_sentence.offset += 1
        bioc_sentence.text = sentence.attrib["text"].strip()
        for entity in sentence.xpath(".//entity"):
            ann = bioc.BioCAnnotation()
            ann.text = entity.attrib["text"]
            start = int(entity.attrib["charOffset"].split(";")[0].split("-")[0])
            end = int(entity.attrib["charOffset"].split(";")[-1].split("-")[1]) + 1
            offset = start + bioc_sentence.offset
            length = end - start
            loc = bioc.BioCLocation(offset=offset, length=length)
            ann.add_location(loc)
            ann.infons["type"] = "CHEMICAL"
            ann.infons["identifier"] = ann.text
            bioc_id = f"T{len(ddi_id_to_bioc_id)+1}"
            ann.id = bioc_id
            ddi_id_to_bioc_id[entity.attrib["id"]] = bioc_id
            bioc_sentence.add_annotation(ann)

        for pair in sentence.xpath(".//pair"):
            if pair.attrib["ddi"] == "false":
                continue
            head = ddi_id_to_bioc_id[pair.attrib["e1"]]
            tail = ddi_id_to_bioc_id[pair.attrib["e2"]]

            rel = pair.attrib.get("type", "INT")

            bioc_rel = bioc.BioCRelation()
            bioc_rel.add_node(bioc.BioCNode(refid=head, role="head"))
            bioc_rel.add_node(bioc.BioCNode(refid=tail, role="tail"))
            bioc_rel.infons["type"] = rel.upper()
            bioc_sentence.add_relation(bioc_rel)

        bioc_passage.add_sentence(bioc_sentence)
        sentence_texts.append(bioc_sentence.text)

    bioc_passage.text = " ".join(sentence_texts)

    # for sentence in bioc_passage.sentences:
    #     for ann in sentence.annotations:
    #         start = ann.locations[0].offset
    #         end = start + ann.locations[0].length
    #         assert bioc_passage.text[start:end] == ann.text


    return bioc_doc



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    collection = bioc.BioCCollection()
    data_dir: Path
    for data_dir in args.input:
        for file in data_dir.rglob("*xml"):
            collection.add_document(get_bioc_from_ddi_file(file))

    args.output: Path
    args.output.parent.mkdir(exist_ok=True)
    with args.output.open("w") as f:
        bioc.dump(collection, f)


