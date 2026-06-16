"""Per-pair batching cost for the score path."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sie_server.core.prepared import ScorePreparedItem

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

SCORE_MEDIA_COST: Final[int] = 1024


def score_item_cost(item: Item) -> int:
    """Return the approximate batching cost for one side of a score pair."""
    text_cost = len(item.text) if item.text else 0
    media_count = len(item.images or [])
    if item.audio is not None:
        media_count += 1
    if item.video is not None:
        media_count += 1
    if item.document is not None:
        media_count += 1
    return text_cost + (media_count * SCORE_MEDIA_COST)


def score_pair_cost(query: Item, doc: Item) -> int:
    """Return the approximate batching cost for a rerank query/document pair."""
    return score_item_cost(query) + score_item_cost(doc)


def build_score_prepared_items(query: Item, items: list[Item]) -> list[ScorePreparedItem]:
    """Build PreparedItems for a score request."""
    return [ScorePreparedItem(cost=score_pair_cost(query, item), original_index=i) for i, item in enumerate(items)]
