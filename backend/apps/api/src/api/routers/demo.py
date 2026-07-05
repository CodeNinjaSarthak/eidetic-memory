"""Read-only demo endpoint: LoCoMo Transcript Explorer.

Runs the eval pipeline (eval/eval_qa_accuracy.py) against the pre-ingested
eidetic_memories Qdrant collection. Never writes to Qdrant — retrieval and
generation only.
"""

import logging
from itertools import zip_longest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.auth import verify_demo_credentials
from api.dependencies import get_demo_retriever, get_llm_service
from api.rate_limit import check_demo_rate_limit
from api.schemas import DemoQueryRequest, DemoQueryResponse, RankedMemory
from llm.generation.base import AbstractLLMService
from retrieval import reranker
from retrieval.context import ContextBuilder
from retrieval.retriever import MemoryRetriever
from storage.models import MemoryFact

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/demo",
    tags=["demo"],
    dependencies=[Depends(verify_demo_credentials)],
)

# --- Eval-matched prompts (verbatim from eval/eval_qa_accuracy.py) ---

ANSWER_SYSTEM_PROMPT = """You are a precise memory retrieval assistant.
Answer the question using ONLY the provided memories.

Rules:
1. Base your answer strictly on the memory text. Do not add any information
   not explicitly stated in the memories.
2. Be concise but complete. If the answer is a single fact, one short sentence.
   If the answer is a list of items, include ALL items from the memories.
3. Pay attention to timestamps. If asked what is current or most recent,
   use the latest memory. Convert relative time references to specific dates
   using memory timestamps.
4. If memories contain contradictory information, use the most recent.
5. If the memories contain ANY relevant information, use it to answer.
   Only say "I don't know" if no memories relate to the question at all.
   Do not say "I don't know" if partial evidence exists — use what you have."""

OPEN_DOMAIN_SYSTEM_PROMPT = """You are a precise memory retrieval assistant.
Answer the question using ONLY the provided memories.

Rules:
1. Base your answer strictly on the memory text. Do not add any information
   not explicitly stated in the memories.
2. Answer directly and specifically. State the key facts clearly first.
   Do not bury the answer in long narrative preamble.
3. Include enough detail to fully answer the question, but do not ramble.
   Two to four sentences is usually sufficient.
4. If memories contain contradictory information, use the most recent.
5. If the memories contain ANY relevant information, use it to answer.
   Only say "I don't know" if no memories relate to the question at all."""

# --- Retrieval constants ---
# Must pass top_k explicitly: fetch_multiplier only fires when jina_api_key is set.
_FETCH_K_PER_SPEAKER = 90   # 30 final * 3 fetch_multiplier
_FINAL_TOP_K = 30
_DISPLAY_TOP_K = 5

_EVAL_USER_ID_PREFIX = "locomo_eval"

# --- LoCoMo conversation map ---
# Speaker names only (sample_id -> speaker_a/speaker_b), extracted from the LoCoMo
# dataset (eval/data/locomo10.json, CC BY-NC 4.0 — not redistributed here). The demo
# only needs these names to build Qdrant user_ids and display labels; the actual
# conversation facts are already ingested into Qdrant and never re-read from the
# raw dataset at runtime. Hardcoded so the app has no file I/O at import time.
_CONV_MAP: dict[str, dict[str, str]] = {
    "conv-26": {"speaker_a": "Caroline", "speaker_b": "Melanie"},
    "conv-30": {"speaker_a": "Jon", "speaker_b": "Gina"},
    "conv-41": {"speaker_a": "John", "speaker_b": "Maria"},
    "conv-42": {"speaker_a": "Joanna", "speaker_b": "Nate"},
    "conv-43": {"speaker_a": "Tim", "speaker_b": "John"},
    "conv-44": {"speaker_a": "Audrey", "speaker_b": "Andrew"},
    "conv-47": {"speaker_a": "James", "speaker_b": "John"},
    "conv-48": {"speaker_a": "Deborah", "speaker_b": "Jolene"},
    "conv-49": {"speaker_a": "Evan", "speaker_b": "Sam"},
    "conv-50": {"speaker_a": "Calvin", "speaker_b": "Dave"},
}

_context_builder = ContextBuilder()


def _make_user_id(conv_id: str, speaker: str) -> str:
    return f"{_EVAL_USER_ID_PREFIX}_{conv_id.replace('-', '_')}_{speaker.lower().replace(' ', '_')}"


def _build_ranked_memories(
    facts: list[MemoryFact],
    position_map: dict[str, int],
    rank_type: str,
) -> list[RankedMemory]:
    """Build RankedMemory list with rank_before from position_map.

    For rank_after (rank_type='after'), rank_after = 1-indexed position in this list.
    For rank_before (rank_type='before'), rank_after = position in position_map if available.
    """
    result = []
    for idx, fact in enumerate(facts[:_DISPLAY_TOP_K], start=1):
        rank_before = position_map.get(fact.content, idx)
        rank_after = idx if rank_type == "after" else None
        result.append(
            RankedMemory(
                content=fact.content,
                user_id=fact.user_id,
                rank_before=rank_before,
                rank_after=rank_after,
            )
        )
    return result


@router.get("/auth-check")
async def demo_auth_check() -> dict[str, bool]:
    """Validate demo credentials without touching the LLM or embeddings.

    Reaching this handler means the router-level Basic Auth dependency already
    accepted the credential, so the frontend can verify a credential at login
    time. Makes zero paid API calls and is deliberately not rate-limited, so a
    credential check never consumes the /demo/query daily budget.
    """
    return {"ok": True}


@router.post(
    "/query",
    response_model=DemoQueryResponse,
    dependencies=[Depends(check_demo_rate_limit)],
)
async def demo_query(
    payload: DemoQueryRequest,
    retriever: Annotated[MemoryRetriever, Depends(get_demo_retriever)],
    llm_service: Annotated[AbstractLLMService, Depends(get_llm_service)],
) -> DemoQueryResponse:
    """Answer a question using memories from a pre-ingested LoCoMo conversation.

    Read-only: calls retriever.retrieve() and llm_service.complete() only.
    Never writes to Qdrant.
    """
    conv = _CONV_MAP.get(payload.conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=422,
            detail=f"conversation_id {payload.conversation_id!r} not found. "
            f"Valid IDs: {sorted(_CONV_MAP.keys())}",
        )

    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    user_id_a = _make_user_id(payload.conversation_id, speaker_a)
    user_id_b = _make_user_id(payload.conversation_id, speaker_b)

    # Retrieve from both speakers — pass top_k explicitly since fetch_multiplier
    # only fires when jina_api_key is set; demo uses local reranker (no Jina).
    memories_a = await retriever.retrieve(
        query=payload.question,
        user_id=user_id_a,
        top_k=_FETCH_K_PER_SPEAKER,
    )
    memories_b = await retriever.retrieve(
        query=payload.question,
        user_id=user_id_b,
        top_k=_FETCH_K_PER_SPEAKER,
    )

    # Round-robin merge with dedup (exact eval logic)
    seen: set[str] = set()
    merged: list[MemoryFact] = []
    for fact in [
        f
        for pair in zip_longest(memories_a, memories_b)
        for f in pair
        if f is not None
    ]:
        if fact.content not in seen:
            seen.add(fact.content)
            merged.append(fact)

    # Build position map for rank_before tracking (1-indexed)
    position_map: dict[str, int] = {f.content: idx for idx, f in enumerate(merged, start=1)}

    # Pre-rerank top-5 (cosine / round-robin order)
    cosine_top5 = merged[:_DISPLAY_TOP_K]
    memories_top5_cosine = _build_ranked_memories(cosine_top5, position_map, rank_type="before")

    # Cross-encoder rerank
    reranked = await reranker.rerank(payload.question, merged, _FINAL_TOP_K)

    # Post-rerank top-5 with rank_before from original merged position
    memories_top5_reranked = _build_ranked_memories(reranked, position_map, rank_type="after")

    # Select prompt and token budget by question_type
    if payload.question_type == "open_domain":
        active_prompt = OPEN_DOMAIN_SYSTEM_PROMPT
        gen_max_tokens = 350
    else:
        active_prompt = ANSWER_SYSTEM_PROMPT
        gen_max_tokens = 200

    # Build context-enriched system prompt and generate answer
    system_prompt = _context_builder.build_system_prompt(active_prompt, reranked)
    messages = [{"role": "user", "content": payload.question}]

    answer = await llm_service.complete(
        messages=messages,
        system=system_prompt,
        temperature=0.0,
        max_tokens=gen_max_tokens,
    )

    two_pass_would_fire = (
        "don't know" in answer.lower() or "do not know" in answer.lower()
    )

    return DemoQueryResponse(
        answer=answer,
        memories_top5_cosine=memories_top5_cosine,
        memories_top5_reranked=memories_top5_reranked,
        two_pass_would_fire=two_pass_would_fire,
        conversation_id=payload.conversation_id,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        question_type=payload.question_type,
    )
