import unittest

import bioc
import numpy as np

from entity_marker_baseline import insert_pair_markers


def test_insert_pair_markers():
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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)
    assert text == "by [HEAD-S][TAIL-S]catechol[HEAD-E]-O[TAIL-E] is"

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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)

    assert text == "by [HEAD-S][TAIL-S]catechol[TAIL-E]-O[HEAD-E] is"

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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)

    assert text == "by [HEAD-S]alpha [TAIL-S]catechol[TAIL-E]-O[HEAD-E] is"

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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)

    assert text == "by [TAIL-S]alpha [HEAD-S]catechol[HEAD-E]-O[TAIL-E] is"

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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)

    assert text == "by [TAIL-S]alpha [HEAD-S]catechol[TAIL-E]-O[HEAD-E] is"


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

    text = insert_pair_markers(text=text, head=head, tail=tail, sentence_offset=0)

    assert text == "by [HEAD-S]foo[HEAD-E] does [TAIL-S]catechol-O[TAIL-E] is"


