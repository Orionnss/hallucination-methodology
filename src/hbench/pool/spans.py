"""Token span of the user text π(q) inside a chat-template prompt (ICR core positions, D69)."""

from __future__ import annotations


def user_token_span(tokenizer, prompt_text: str, prompt_ids: list[int], user_text: str) -> tuple[int, int]:
    """[start, end) token positions of `user_text` in the unpadded prompt. Uses the tokenizer's offset
    mapping on the prompt text, and checks that this tokenization equals the saved prompt IDs (D7)."""
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(prompt_ids):
        raise ValueError("re-tokenised prompt text differs from the saved prompt IDs")
    c0 = prompt_text.rfind(user_text)
    if c0 < 0:
        raise ValueError("user text not found in the prompt text")
    c1 = c0 + len(user_text)
    toks = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < c1 and b > c0]
    if not toks:
        raise ValueError("no token overlaps the user text")
    return toks[0], toks[-1] + 1
