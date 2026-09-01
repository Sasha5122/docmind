"""Reciprocal Rank Fusion is pure arithmetic, so most of this runs without a database."""

import pytest

from docmind.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from docmind.retrieval.search import RetrievedChunk


def chunk(chunk_id: int, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        filename="doc.pdf",
        title=None,
        page=1,
        lang="de",
        text=f"chunk {chunk_id}",
        score=score,
    )


def test_item_in_both_lists_beats_items_in_one() -> None:
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword = [chunk(3), chunk(4)]
    fused = reciprocal_rank_fusion([vector, keyword])
    assert [c.chunk_id for c in fused] == [3, 1, 2, 4]
    # rank 3 in the first list + rank 1 in the second
    assert fused[0].score == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))


def test_single_list_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([[chunk(5), chunk(6), chunk(7)]])
    assert [c.chunk_id for c in fused] == [5, 6, 7]


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_ties_break_deterministically_by_chunk_id() -> None:
    fused = reciprocal_rank_fusion([[chunk(9)], [chunk(2)]])
    assert [c.chunk_id for c in fused] == [2, 9]


def test_original_scores_are_replaced_by_rrf_scores() -> None:
    fused = reciprocal_rank_fusion([[chunk(1, score=0.99)]])
    assert fused[0].score == pytest.approx(1 / (RRF_K + 1))
    assert fused[0].text == "chunk 1"
