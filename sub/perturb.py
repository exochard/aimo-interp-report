"""Deterministic, math-preserving prompt variants.

Only prose changes: text inside $...$, \\(...\\), \\[...\\] and every digit stay byte for byte,
and a variant whose digit sequence or math spans differ from the original is rejected (the
original is used in its place and the rejection is reported). The organisers' variants
(typos, distract, rename, rephrase, domain, expert) are unknown; these are proxies.
"""

import hashlib
import random
import re

MATH = re.compile(r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
NEUTRAL = (
    "The problem was first posed in a regional mathematics circle.",
    "Take your time and read every condition carefully.",
    "This question appeared in a practice set for a competition.",
)


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def _math_and_digits(text: str) -> tuple:
    return tuple(MATH.findall(text)), re.sub(r"\D", "", text)


def typos(text: str) -> str:
    """Swap two inner letters in about one prose word in eight."""
    rng = random.Random(_seed(text))
    out = []
    for part in MATH.split(text):
        if MATH.fullmatch(part):
            out.append(part)
            continue
        words = part.split(" ")
        for i, w in enumerate(words):
            if len(w) > 4 and w.isalpha() and rng.random() < 0.125:
                j = rng.randrange(1, len(w) - 2)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        out.append(" ".join(words))
    return "".join(out)


def distract(text: str) -> str:
    """Append one irrelevant sentence that carries no number."""
    return f"{text.rstrip()} {NEUTRAL[_seed(text) % len(NEUTRAL)]}"


VARIANTS = {"typos": typos, "distract": distract}


def variants(text: str) -> tuple:
    """({name: text}, [rejected names])."""
    want = _math_and_digits(text)
    out, rejected = {}, []
    for name, fn in VARIANTS.items():
        v = fn(text)
        if _math_and_digits(v) != want:
            rejected.append(name)
            v = text
        out[name] = v
    return out, rejected
