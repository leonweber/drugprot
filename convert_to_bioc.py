import argparse
from collections import defaultdict
from pathlib import Path

from flair.tokenization import SegtokSentenceSplitter
from pedl.utils import DataGetter
import bioc
from sklearn.model_selection import train_test_split

TYPE_MAP = {
    "CHEMICAL": "Chemical",
    "GENE-Y": "Gene",
    "GENE-N": "Gene"
}


def match_to_pubtator(ann: bioc.BioCAnnotation, pubtator_doc: bioc.BioCDocument,
                      entity_type: str):
    for passage in pubtator_doc.passages:
        for ann_pubtator in passage.annotations:
            if (ann_pubtator.infons["type"] == entity_type and
                (
                        ann_pubtator.locations[0].offset + passage.offset == ann.locations[0].offset
                        or ann_pubtator.text == ann.text
                )
            ):
                return ann_pubtator

    return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--abstracts", type=Path, required=True)
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--relations", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pubtator", help="enrich with pubtator annotations",
                        action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    sentence_splitter = SegtokSentenceSplitter()
    pmid_to_sentences = {}
    pmid_to_text = {}
    pmid_to_entities = defaultdict(dict)
    pmid_to_relations = defaultdict(set)
    data_getter = DataGetter(set(), skip_gene2pmid=True)
    collection = bioc.BioCCollection()

    with open(args.abstracts) as f:
        for line in f:
            fields = line.strip().split("\t")
            pmid, title, abstract = fields
            pmid_to_text[pmid] = title + "\t" + abstract
            title_sentences = sentence_splitter.split(title)
            abstract_sentences = sentence_splitter.split(abstract)
            for sent in abstract_sentences:
                sent.start_pos += len(title) + 1
                sent.end_pos += len(title) + 1
            pmid_to_sentences[pmid] = title_sentences + abstract_sentences

    with open(args.entities) as f:
        for line in f:
            fields = line.strip().split("\t")
            pmid, ent_id, ent_type, start, end, mention = fields
            pmid_to_entities[pmid][ent_id] = (
                ent_type, int(start), int(end), mention)

    with open(args.relations) as f:
        for line in f:
            fields = line.strip().split("\t")
            pmid, rel_type, arg1, arg2 = fields
            ent1 = arg1.split(":")[1]
            ent2 = arg2.split(":")[1]
            pmid_to_relations[pmid].add((rel_type, ent1, ent2))

    pubtator_annotations = {}
    if args.pubtator:
        for doc_chunk in data_getter.get_documents_from_api(sorted(pmid_to_text)):
            for doc in doc_chunk:
                pubtator_annotations[doc.id] = doc

    n_relations_in_sentence = 0
    n_relations_total = 0
    n_entities_normalized = 0
    n_entities_total = 0
    for pmid in sorted(pmid_to_text):
        doc = bioc.BioCDocument()
        collection.add_document(doc)
        pubtator_doc = pubtator_annotations.get(pmid)
        doc.id = pmid
        if args.pubtator:
            doc.infons["has_pubtator"] = pubtator_doc is not None

        passage = bioc.BioCPassage()
        doc.add_passage(passage)
        passage.offset = 0
        passage.text = pmid_to_text[pmid]

        annotations = []
        for entity_id, entity in pmid_to_entities[pmid].items():
            annotation = bioc.BioCAnnotation()
            annotation.id = entity_id
            annotation.text = entity[3]
            annotation.infons["type"] = entity[0]
            annotation.add_location(bioc.BioCLocation(entity[1], entity[2]-entity[1]))
            if pubtator_doc:
                pubtator_annotation = match_to_pubtator(annotation, pubtator_doc,
                                                        entity_type=TYPE_MAP[entity[0]])
            else:
                pubtator_annotation = False

            if pubtator_annotation:
                for k, v in list(pubtator_annotation.infons.items()):
                    pubtator_annotation.infons[k.lower()] = v
                annotation.infons["identifier"] = pubtator_annotation.infons["identifier"]
                n_entities_normalized += 1
            else:
                annotation.infons["identifier"] = annotation.text.lower()
            n_entities_total += 1

            annotations.append(annotation)

        n_relations_total += len(pmid_to_relations[pmid])
        for sentence in pmid_to_sentences[pmid]:
            bioc_sentence = bioc.BioCSentence()
            passage.add_sentence(bioc_sentence)
            bioc_sentence.text = sentence.to_original_text()
            bioc_sentence.offset = sentence.start_pos
            sentence_annotations = [i for i in annotations if sentence.start_pos <= i.locations[0].offset <= sentence.end_pos]
            sentence_entities = set(i.id for i in sentence_annotations)
            bioc_sentence.annotations = sentence_annotations

            sentence_relations = [i for i in pmid_to_relations[pmid] if {i[1], i[2]} <= sentence_entities]
            n_relations_in_sentence += len(sentence_relations)
            for rel in sentence_relations:
                bioc_rel = bioc.BioCRelation()
                bioc_rel.add_node(bioc.BioCNode(refid=rel[1], role="head"))
                bioc_rel.add_node(bioc.BioCNode(refid=rel[2], role="tail"))
                bioc_rel.infons["type"] = rel[0]
                bioc_sentence.add_relation(bioc_rel)


    with open(args.out, "w") as f:
        bioc.dump(collection, f)
        print(f"Wrote {len(collection.documents)} docs to {args.out}")

    print(f"{100*n_entities_normalized/n_entities_total:.2f}% of entities could be normalized")
    print(f"{100*n_relations_in_sentence/n_relations_total:.2f}% of relations are intra-sentential")

