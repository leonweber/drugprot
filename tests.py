import unittest

import bioc
import numpy as np

from drugprot.models.entity_marker_baseline import insert_pair_markers


def test_insert_pair_markers_special_tokens():
    text = "by catechol-O is"
    head_mention = "catechol"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)
    assert text == "by <e1><e2>catechol</e1>-O</e2> is"

    text = "by catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)

    assert text == "by <e1><e2>catechol</e2>-O</e1> is"

    text = "by alpha catechol-O is"
    head_mention = "alpha catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)

    assert text == "by <e1>alpha <e2>catechol</e2>-O</e1> is"

    text = "by alpha catechol-O is"
    head_mention = "catechol"
    tail_mention = "alpha catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)

    assert text == "by <e2>alpha <e1>catechol</e1>-O</e2> is"

    text = "by alpha catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "alpha catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)

    assert text == "by <e2>alpha <e1>catechol</e2>-O</e1> is"


    text = "by foo does catechol-O is"
    head_mention = "foo"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=True, blind_entities=False)

    assert text == "by <e1>foo</e1> does <e2>catechol-O</e2> is"

def test_insert_pair_markers_single_chars():
    text = "by catechol-O is"
    head_mention = "catechol"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)
    assert text == "by @@catechol$-O$ is"

    text = "by catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)

    assert text == "by @@catechol$-O$ is"

    text = "by alpha catechol-O is"
    head_mention = "alpha catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)

    assert text == "by @alpha @catechol$-O$ is"

    text = "by alpha catechol-O is"
    head_mention = "catechol"
    tail_mention = "alpha catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)

    assert text == "by @alpha @catechol$-O$ is"

    text = "by alpha catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "alpha catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)

    assert text == "by @alpha @catechol$-O$ is"


    text = "by foo does catechol-O is"
    head_mention = "foo"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=False)

    assert text == "by @foo$ does @catechol-O$ is"


def test_insert_pair_markers_entity_blinding():
    text = "by catechol-O is"
    head_mention = "catechol"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)
    assert text == "by @@HEAD-TAIL$$ is"

    text = "by catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)

    assert text == "by @@HEAD-TAIL$$ is"

    text = "by alpha catechol-O is"
    head_mention = "alpha catechol-O"
    tail_mention = "catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)

    assert text == "by @@HEAD-TAIL$$ is"

    text = "by alpha catechol-O is"
    head_mention = "catechol"
    tail_mention = "alpha catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)

    assert text == "by @@HEAD-TAIL$$ is"

    text = "by alpha catechol-O is"
    head_mention = "catechol-O"
    tail_mention = "alpha catechol"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)

    assert text == "by @@HEAD-TAIL$$ is"


    text = "by foo does catechol-O is"
    head_mention = "foo"
    tail_mention = "catechol-O"
    head = bioc.BioCAnnotation()
    head.text = head_mention
    head.add_location(bioc.BioCLocation(offset=text.index(head_mention),
                                        length=len(head_mention)))

    tail = bioc.BioCAnnotation()
    tail.text = tail_mention
    tail.add_location(bioc.BioCLocation(offset=text.index(tail_mention),
                                        length=len(tail_mention)))

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0,
                               mark_with_special_tokens=False, blind_entities=True)

    assert text == "by @HEAD$ does @TAIL$ is"

