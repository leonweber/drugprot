from operator import itemgetter

import bioc
import pandas as pd


def get_drugprot_df():
    df = {"head": [], "rel": [], "tail": []}

    collection = bioc.load("data/drugprot_biosyn_norm/train.bioc.xml")
    for document in collection.documents:
        for passage in document.passages:
            for sentence in passage.sentences:
                ent_id_to_cuid = {}
                for ann in sentence.annotations:
                    ent_id_to_cuid[ann.id] = ann.infons["identifier"]

                for rel in sentence.relations:
                    head = rel.get_node("head").refid
                    tail = rel.get_node("tail").refid
                    rel = rel.infons["type"]

                    for head_cuid in ent_id_to_cuid[head].split("|"):
                        for tail_cuid in ent_id_to_cuid[tail].split("|"):
                            head_cuid = "MESH:" + head_cuid

                            df["head"].append(head_cuid)
                            df["rel"].append(rel)
                            df["tail"].append(tail_cuid)

    return pd.DataFrame(df)

def check_alignment(src, tgt, topk=5):

    src_rels = src.rel.unique()
    tgt_rels = tgt.rel.unique()

    for tgt_rel in tgt_rels:
        tgt_mask = tgt.rel == tgt_rel
        tgt_pairs = set(zip(tgt[tgt_mask]["head"].astype(str), tgt[tgt_mask]["tail"].astype(str)))
        alignments = []
        for src_rel in src_rels:

            src_mask = src.rel == src_rel
            src_pairs = set(zip(src[src_mask]["head"].astype(str), src[src_mask]["tail"].astype(str)))

            tp = len(src_pairs & tgt_pairs)
            fn = len(tgt_pairs - src_pairs)
            fp = len(src_pairs - tgt_pairs)


            try:
                f1 = tp / (tp + 0.5 * (fp + fn))
            except ZeroDivisionError:
                f1 = 0

            alignments.append((src_rel, f1))
        topk_alignments = sorted(alignments, key=itemgetter(1), reverse=True)[:topk]
        print(tgt_rel, topk_alignments)
        print()




if __name__ == '__main__':
    drugprot = get_drugprot_df()
    ttd = pd.read_csv("data/ttd.tsv", header=None, sep="\t")
    ttd.columns = ["head", "rel", "tail"]
    check_alignment(ttd, drugprot)

