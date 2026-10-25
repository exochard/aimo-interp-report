"""A small fixed family of equivalence-preserving problem perturbations, each with a check.

Unlike perturb.py (prose-only proxies), these edit the mathematics in ways that provably keep
the answer: a variable renamed, a literal constant re-expressed, independent givens
reordered, a ratio or fraction restated. Eight variants in seven families (a family is the
unit of coverage: two renames or two constant rewrites are near-duplicates, one family):

- rename_a, rename_b: one single-letter variable (lower case, or an isolated capital) renamed
  to an unused letter everywhere; the article "a" in prose does not block a variable "a";
- const_sum, const_diff: up to three literal integers written as (a+b) or (a+d-d)-style sums;
- exponent: a literal integer exponent written as a sum, x^{2} as x^{1+1};
- commute: "p+q" with plain operands written "q+p";
- flip: a comparison "b>9" written "9<b";
- reorder: assignment givens ("AB = 20, R = 26"), align rows or "$m$ and $n$ are relatively
  prime" exchanged in order;
- restate: "a:b" written as \\frac{a}{b}, and \\frac{p}{q} with a terminating decimal written
  as that decimal.

Every variant carries a deterministic validity check, recomputed from the original and the
variant alone (verify): "sympy" (each edited expression is parsed and compared exactly),
"roundtrip" (the inverse rename restores the original byte for byte) or "audit" (the edit
is structurally a permutation but its order-independence is a reading of the sentence, so a
human should look). A tikzpicture block is opaque: nothing inside it is touched.

  uv run --no-project --with pandas --with pyarrow --with sympy python sub/invariance.py
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from fractions import Fraction

import sympy

TIKZ = re.compile(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", re.S)
MATH = re.compile(
    r"(\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{(align\*?|equation\*?)\}.*?\\end\{\2\})", re.S)
OPAQUE_ARG = re.compile(r"\\(?:text|textbf|mathrm|mathbb|mathbf|mathcal|operatorname|begin|end)\{[^}]*\}")
COMMAND = re.compile(r"\\[A-Za-z]+")
FUNCTION_WORDS = re.compile(r"(?<![A-Za-z\\])(?:sin|cos|tan|log|ln|gcd|lcm|max|min|det|mod|lim|sup|inf)(?![A-Za-z])")
CANDIDATES = "nmkrsfxyzpqabc"
UPPER_CANDIDATES = "NSMRKFPTL"
POOLS = {"rename_a": "tuwvhg", "rename_b": "ghwvut"}
UPPER_POOLS = {"rename_a": "UVWHGQ", "rename_b": "QGHWVU"}
NUMBER = re.compile(r"(?<![\w.^_{}\\])(\d+)(?![\w^_{}]|\.\d)")
EXPONENT = re.compile(r"\^(?:\{(\d{1,6})\}|(\d))")
ATOM = r"(?:[A-Za-z]|\d+)"
COMMUTE = re.compile(r"(?<![\w\\.])(" + ATOM + r")\s*\+\s*(" + ATOM + r")(?![\w])")
FLIP = re.compile(r"(?<![\w\\.])(" + ATOM + r")\s*([<>])\s*(" + ATOM + r")(?![\w])")
# A swap next to these characters would change precedence or bind a neighbouring operator.
SUM_BEFORE_BAD, SUM_AFTER_BAD = set("^_*/+-!|\\"), set("^_*/!'(|\\")
REL_BEFORE_BAD, REL_AFTER_BAD = set("^_*/+-!|\\<>=("), set("^_*/+-!'(|\\<>=")
ARTICLE_NOT = {"and", "or", "is", "are", "be", "then", "such", "to", "with", "if", "for", "so"}
RATIO = re.compile(r"(?<![\w.:])(\d+):(\d+)(?![\w.:])")
FRAC = re.compile(r"(?<!\d)\\frac\{(\d+)\}\{(\d+)\}")
ASSIGN = re.compile(r"[A-Za-z]{1,3}(?:_\{?\w{1,3}\}?)?\s*=\s*-?\d+(?:\.\d+)?")
ASSIGN_SPAN = re.compile(r"\$\s*(" + ASSIGN.pattern + r")\s*[.,;]?\s*\$")
JOIN = re.compile(r",\s*(?:and\s+)?|\s+and\s+")
PAIR = re.compile(r"\$([A-Za-z])\$ and \$([A-Za-z])\$(?= are relatively prime)|both \$(\w+)\$ and \$(\w+)\$")
ALIGN = re.compile(r"\\begin\{(align\*?)\}(.*?)\\end\{\1\}", re.S)


def _seed(text: str, salt: str = "") -> int:
    return int.from_bytes(hashlib.sha256((salt + text).encode()).digest()[:4], "little")


@dataclass(frozen=True)
class Perturbation:
    name: str
    text: str
    check: str  # "sympy" | "roundtrip" | "audit"
    meta: dict = field(default_factory=dict, compare=False, hash=False)
    valid: bool = False


def _spans(text: str) -> list:
    """(start, end) of every math span outside tikz blocks."""
    opaque = [m.span() for m in TIKZ.finditer(text)]
    return [m.span() for m in MATH.finditer(text) if not any(a <= m.start() < b for a, b in opaque)]


def _prose(text: str) -> str:
    out, last = [], 0
    for a, b in _spans(text):
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return " ".join(out)


# ---- rename -----------------------------------------------------------------------------

def _rename_math(span: str, old: str, new: str) -> str:
    """Rename the single-letter variable `old` in one math span, leaving commands and the
    arguments of font and text commands alone. A capital is renamed only where it stands
    alone (not inside ABC or AB), a lower-case letter wherever it occurs."""
    keep = []

    def stash(m):
        keep.append(m.group(0))
        return f"\0{len(keep) - 1}\0"

    s = COMMAND.sub(stash, OPAQUE_ARG.sub(stash, span))
    s = re.sub(rf"(?<![A-Za-z]){old}(?![A-Za-z])", new, s) if old.isupper() else s.replace(old, new)
    return re.sub(r"\0(\d+)\0", lambda m: keep[int(m.group(1))], s)


def _rename_text(text: str, old: str, new: str) -> str:
    out, last = [], 0
    for a, b in _spans(text):
        out.append(text[last:a])
        out.append(_rename_math(text[a:b], old, new))
        last = b
    out.append(text[last:])
    return "".join(out)


def _letters_in_math(text: str) -> set:
    """Variable letters, one per character, after commands and font arguments are removed."""
    found = set()
    for a, b in _spans(text):
        s = COMMAND.sub(" ", OPAQUE_ARG.sub(" ", text[a:b]))
        found |= set(re.findall(r"[A-Za-z]", s))
    return found


def _isolated_capitals(text: str) -> set:
    """Capitals that occur somewhere on their own (a scalar such as $N$), not only inside ABC."""
    found = set()
    for a, b in _spans(text):
        s = COMMAND.sub(" ", OPAQUE_ARG.sub(" ", text[a:b]))
        found |= set(re.findall(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", s))
    return found


def _prose_word(prose: str, c: str) -> bool:
    """`c` stands in prose as a word and is a variable there. The article ("a river", "A real
    number") is not: a lone a/A followed by a lower-case word other than a connective."""
    for m in re.finditer(rf"(?<![A-Za-z]){c}(?![A-Za-z])", prose):
        if c in "aA":
            nxt = re.match(r"\s+([a-z]+)", prose[m.end():])
            if nxt and nxt.group(1) not in ARTICLE_NOT:
                continue
        return True
    return False


def _free_letter(text: str, letter: str) -> bool:
    """`letter` appears nowhere as a variable and nowhere as a prose word."""
    return letter not in _letters_in_math(text) and not _prose_word(_prose(text), letter)


def rename(text: str, name: str) -> Perturbation | None:
    if TIKZ.search(text) or any(FUNCTION_WORDS.search(text[a:b]) for a, b in _spans(text)):
        return None
    prose = _prose(text)
    letters, caps = _letters_in_math(text), _isolated_capitals(text)
    usable = [c for c in CANDIDATES if c in letters and not _prose_word(prose, c)]
    usable += [c for c in UPPER_CANDIDATES if c in caps and not _prose_word(prose, c)]
    if not usable:
        return None
    old = usable[_seed(text, name) % len(usable)]
    new = next((c for c in (UPPER_POOLS if old.isupper() else POOLS)[name] if _free_letter(text, c)), None)
    if new is None:
        return None
    return Perturbation(name, _rename_text(text, old, new), "roundtrip", {"from": old, "to": new})


# ---- edits (constants, restatements) ----------------------------------------------------

def _apply_edits(text: str, edits: list) -> str:
    """edits: (start, end, replacement) over the original; applied from the end."""
    for a, b, new in sorted(edits, reverse=True):
        text = text[:a] + new + text[b:]
    return text


def _expr(s: str):
    """Parse a restated expression: sums, differences, decimals, a:b, \\frac{a}{b}, an exponent
    token (^{1+1}), a sum of plain letters, or a comparison of two atoms."""
    s = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"((\1)/(\2))", s).replace(":", "/")
    s = s.lstrip("^").strip("{}")
    local = {c: sympy.Symbol(c) for c in set(re.findall(r"[A-Za-z]", s))}
    return sympy.sympify(s, locals=local, rational=True)


def _number_sites(text: str) -> list:
    sites = []
    for a, b in _spans(text):
        span = text[a:b]
        sites += [(a + m.start(), a + m.end(), int(m.group(1))) for m in NUMBER.finditer(span) if int(m.group(1)) >= 2]
        for m in FRAC.finditer(span):  # numerator and denominator are plain integers
            sites += [(a + m.start(g), a + m.end(g), int(m.group(g))) for g in (1, 2) if int(m.group(g)) >= 2]
    return sorted(set(sites))


def constants(text: str, name: str) -> Perturbation | None:
    sites = _number_sites(text)
    if not sites:
        return None
    rng = random.Random(_seed(text, name))
    chosen = sorted(rng.sample(sites, min(3, len(sites))))
    edits = []
    for a, b, v in chosen:
        if name == "const_sum":
            lo = rng.randint(1, v - 1)
            new = f"({v - lo}+{lo})"
        else:
            d = rng.randint(1, 9)
            new = f"({v + d}-{d})"
        edits.append((a, b, new))
    return Perturbation(name, _apply_edits(text, edits), "sympy", {"edits": [(a, b, text[a:b], n) for a, b, n in edits]})


def restate(text: str) -> Perturbation | None:
    edits = []
    for a, b in _spans(text):
        span = text[a:b]
        for m in RATIO.finditer(span):
            edits.append((a + m.start(), a + m.end(), f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"))
        for m in FRAC.finditer(span):
            p, q = int(m.group(1)), int(m.group(2))
            f = Fraction(p, q)
            den = f.denominator
            while den % 2 == 0:
                den //= 2
            while den % 5 == 0:
                den //= 5
            if den == 1 and f.denominator > 1:
                edits.append((a + m.start(), a + m.end(), format(sympy.Rational(p, q).evalf(30).round(12), "f").rstrip("0").rstrip(".")))
    if not edits:
        return None
    return Perturbation("restate", _apply_edits(text, edits), "sympy", {"edits": [(a, b, text[a:b], n) for a, b, n in edits]})


def _edit_meta(text, edits):
    return {"edits": [(a, b, text[a:b], n) for a, b, n in edits]}


def _pick(text: str, name: str, edits: list, limit: int = 3) -> list:
    if len(edits) <= limit:
        return sorted(edits)
    return sorted(random.Random(_seed(text, name)).sample(edits, limit))


def exponent(text: str) -> Perturbation | None:
    """A literal integer exponent written as a sum: x^{2} as x^{1+1}, 10^{101} as 10^{100+1}."""
    edits = []
    for a, b in _spans(text):
        for m in EXPONENT.finditer(text[a:b]):
            v = int(m.group(1) or m.group(2))
            if v >= 2:
                lo = random.Random(_seed(text, f"exp{a + m.start()}")).randint(1, v - 1)
                edits.append((a + m.start(), a + m.end(), f"^{{{v - lo}+{lo}}}"))
    if not edits:
        return None
    edits = _pick(text, "exponent", edits)
    return Perturbation("exponent", _apply_edits(text, edits), "sympy", _edit_meta(text, edits))


def _neighbours_ok(text: str, a: int, b: int, before_bad: set, after_bad: set) -> bool:
    pre, post = text[:a].rstrip(), text[b:].lstrip()
    if pre and pre[-1] in before_bad:
        return False
    if post and (post[0] in after_bad or (post[0] == "." and post[1:2].isdigit())):
        return False
    return True


def commute(text: str) -> Perturbation | None:
    """p+q written q+p, for two plain operands that no neighbouring operator binds."""
    edits = []
    for a, b in _spans(text):
        for m in COMMUTE.finditer(text[a:b]):
            s0, e0 = a + m.start(), a + m.end()
            if _neighbours_ok(text, s0, e0, SUM_BEFORE_BAD, SUM_AFTER_BAD) and m.group(1) != m.group(2):
                edits.append((s0, e0, text[a + m.start(2):a + m.end(2)] + text[a + m.end(1):a + m.start(2)] + text[a + m.start(1):a + m.end(1)]))
    if not edits:
        return None
    edits = _pick(text, "commute", edits)
    return Perturbation("commute", _apply_edits(text, edits), "sympy", _edit_meta(text, edits))


def flip(text: str) -> Perturbation | None:
    """A comparison written the other way round: b>9 as 9<b."""
    edits = []
    for a, b in _spans(text):
        for m in FLIP.finditer(text[a:b]):
            s0, e0 = a + m.start(), a + m.end()
            if _neighbours_ok(text, s0, e0, REL_BEFORE_BAD, REL_AFTER_BAD) and m.group(1) != m.group(3):
                edits.append((s0, e0, f"{m.group(3)}{'<' if m.group(2) == '>' else '>'}{m.group(1)}"))
    if not edits:
        return None
    edits = _pick(text, "flip", edits)
    return Perturbation("flip", _apply_edits(text, edits), "sympy", _edit_meta(text, edits))


def eqswap(text: str) -> Perturbation | None:
    """A lone equality written the other way round: "AB = BC" as "BC = AB"."""
    edits = []
    for a, b in _spans(text):
        span = text[a:b]
        if not span.startswith("$") or span.startswith("$$"):
            continue
        body = span[1:-1]
        core = body.rstrip(" .;,")
        lead = len(body) - len(body.lstrip())
        sides = _sides(core.strip())
        if sides and sides[0] != sides[1]:
            edits.append((a + 1 + lead, a + 1 + len(core), f"{sides[1]} = {sides[0]}"))
    if not edits:
        return None
    edits = _pick(text, "eqswap", edits)
    return Perturbation("eqswap", _apply_edits(text, edits), "sympy", _edit_meta(text, edits))


# ---- surface notation (no mathematics changes; roundtrip-checked) -----------------------

def delimiter(text: str) -> Perturbation | None:
    """Inline math written \\( ... \\) in place of $ ... $."""
    if "\\(" in text or "\\)" in text:
        return None
    edits = [(a, b, "\\(" + text[a + 1:b - 1] + "\\)") for a, b in _spans(text)
             if text[a] == "$" and not text.startswith("$$", a)]
    if not edits:
        return None
    return Perturbation("delimiter", _apply_edits(text, edits), "roundtrip", {"inverse": "delimiter"})


def paren(text: str) -> Perturbation | None:
    """Balanced parentheses written \\left( ... \\right) in spans with no brackets."""
    if "\\left" in text or "\\right" in text:
        return None
    edits = []
    for a, b in _spans(text):
        span = text[a:b]
        if "[" in span or "]" in span or "\\(" in span or "\\)" in span or "\\{" in span:
            continue
        depth, ok, found = 0, True, []
        for i, c in enumerate(span):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                ok = ok and depth >= 0
            if c in "()":
                found.append((a + i, c))
        if ok and depth == 0 and found:
            edits += [(i, i + 1, "\\left(" if c == "(" else "\\right)") for i, c in found]
    if not edits:
        return None
    return Perturbation("paren", _apply_edits(text, edits), "roundtrip", {"inverse": "paren"})


def _undo_surface(variant: str, kind: str) -> str:
    if kind == "paren":
        return variant.replace("\\left(", "(").replace("\\right)", ")")
    out, last = [], 0
    for a, b in _spans(variant):
        out.append(variant[last:a])
        out.append("$" + variant[a + 2:b - 2] + "$" if variant.startswith("\\(", a) else variant[a:b])
        last = b
    out.append(variant[last:])
    return "".join(out)


# ---- reorder ----------------------------------------------------------------------------

def _groups(text: str) -> list:
    """Groups of exchangeable regions: (kind, [(start, end), ...]) in text order."""
    groups, taken = [], []
    opaque = [m.span() for m in TIKZ.finditer(text)]
    inside = lambda i: any(a <= i < b for a, b in opaque)  # noqa: E731
    for a, b in _spans(text):
        span = text[a:b]
        if span.startswith("$") and not span.startswith("$$"):
            body = span[1:-1]
            parts, pos = [], 0
            for m in JOIN.finditer(body):
                parts.append((pos, m.start()))
                pos = m.end()
            if body[pos:].strip(" .;,"):
                parts.append((pos, len(body)))
            regions = []
            for s, e in parts:
                core = ASSIGN.match(body, s, e)
                if core is None or body[core.end():e].strip(" .;,"):
                    regions = []
                    break
                regions.append((a + 1 + core.start(), a + 1 + core.end()))
            if len(regions) >= 2:
                groups.append(("assign", regions))
                taken.append((a, b))
        elif span.startswith("\\begin{align"):
            m = ALIGN.match(span)
            body, off = m.group(2), m.start(2)
            rows, pos = [], 0
            for r in re.finditer(r"\\\\", body + "\\\\"):
                row = body[pos:r.start()]
                if row.strip():
                    lead = len(row) - len(row.lstrip())
                    rows.append((a + off + pos + lead, a + off + pos + len(row.rstrip())))
                pos = r.end()
            if len(rows) >= 2 and all(text[s:e].count("=") == 1 for s, e in rows):
                groups.append(("align", rows))
    # Runs of single-assignment spans joined by commas / "and" in prose.
    atoms = [(m.start(1), m.end(1), m.start(), m.end()) for m in ASSIGN_SPAN.finditer(text)
             if not inside(m.start()) and not any(a <= m.start() < b for a, b in taken)]
    run = []
    for atom in atoms + [None]:
        if atom and run and JOIN.fullmatch(text[run[-1][3]:atom[2]]):
            run.append(atom)
            continue
        if len(run) >= 2:
            groups.append(("assign", [(s, e) for s, e, _, _ in run]))
        run = [atom] if atom else []
    for m in PAIR.finditer(text):
        if not inside(m.start()):
            g = (1, 2) if m.group(1) else (3, 4)
            groups.append(("pair", [(m.start(g[0]), m.end(g[0])), (m.start(g[1]), m.end(g[1]))]))
    return sorted(groups, key=lambda g: g[1][0])


def reorder(text: str) -> Perturbation | None:
    groups = _groups(text)
    if not groups:
        return None
    edits = []
    for _, regions in groups:
        cores = [text[s:e] for s, e in regions]
        if cores == cores[::-1]:
            continue
        edits += [(s, e, c) for (s, e), c in zip(regions, cores[::-1])]
    if not edits:
        return None
    check = "sympy" if all(k == "assign" for k, _ in groups) else "audit"
    return Perturbation("reorder", _apply_edits(text, edits), check, {"kinds": sorted({k for k, _ in groups})})


# ---- validity ---------------------------------------------------------------------------

def _assignment(core: str) -> tuple:
    lhs, rhs = core.split("=")
    return sympy.Symbol(lhs.strip()), sympy.Rational(rhs.strip())


def _sides(eq: str):
    """(lhs, rhs) of a pure equality with one top-level "=", else None."""
    depth, at = 0, []
    for i, c in enumerate(eq):
        depth += c in "({[" or 0
        depth -= c in ")}]" or 0
        if c == "=" and depth == 0:
            at.append(i)
    if len(at) != 1 or re.search(r"[<>,;&]|\\(?:[lg]e|neq?|approx|equiv|colon|to|begin|\\)|\\\\", eq):
        return None
    lhs, rhs = eq[:at[0]].strip(), eq[at[0] + 1:].strip()
    return (lhs, rhs) if lhs and rhs else None


def _same(old: str, new: str) -> bool:
    if "=" in old and "=" in new and _sides(old) and _sides(new):
        return _sides(old) == _sides(new)[::-1]
    x, y = _expr(old), _expr(new)
    if isinstance(x, sympy.core.relational.Relational) or isinstance(y, sympy.core.relational.Relational):
        return x.canonical == y.canonical
    return sympy.simplify(x - y) == 0


def verify(original: str, p: Perturbation) -> bool:
    """Recompute validity from the original text and the variant alone."""
    if p.text == original or not p.text.strip():
        return False
    try:
        if p.name.startswith("rename"):
            old, new = p.meta["from"], p.meta["to"]
            return (_free_letter(original, new) and _rename_text(p.text, new, old) == original
                    and TIKZ.findall(p.text) == TIKZ.findall(original))
        if p.name.startswith("const") or p.name in ("restate", "exponent", "commute", "flip", "eqswap"):
            rebuilt = p.text
            for a, b, old, new in sorted(p.meta["edits"], reverse=True):
                # positions refer to the original; map each edit into the variant by offset.
                shift = sum(len(n) - (e - s) for s, e, _, n in p.meta["edits"] if s < a)
                if p.text[a + shift:a + shift + len(new)] != new or original[a:b] != old:
                    return False
                if not _same(old, new):
                    return False
                rebuilt = rebuilt[:a + shift] + old + rebuilt[a + shift + len(new):]
            return rebuilt == original
        if p.name in ("delimiter", "paren"):
            return (p.name != "paren" or not ("\\left" in original or "\\right" in original)) and \
                _undo_surface(p.text, p.name) == original and TIKZ.findall(p.text) == TIKZ.findall(original) and \
                not ("\\(" in original or "\\)" in original)
        if p.name == "reorder":
            g0, g1 = _groups(original), _groups(p.text)
            if [k for k, _ in g0] != [k for k, _ in g1] or len(g0) == 0:
                return False
            mask = lambda t, gs: _apply_edits(t, [(s, e, "\0") for _, rs in gs for s, e in rs])  # noqa: E731
            if mask(original, g0) != mask(p.text, g1):
                return False
            for (kind, r0), (_, r1) in zip(g0, g1):
                c0, c1 = [original[s:e] for s, e in r0], [p.text[s:e] for s, e in r1]
                if sorted(c0) != sorted(c1):
                    return False
                if kind == "assign" and {_assignment(c) for c in c0} != {_assignment(c) for c in c1}:
                    return False
            return True
    except (sympy.SympifyError, ValueError, KeyError, TypeError):
        return False
    return False


def perturbations(text: str) -> list:
    """Every applicable variant that passes its check, original text excluded, texts distinct."""
    made = [rename(text, "rename_a"), rename(text, "rename_b"), constants(text, "const_sum"),
            constants(text, "const_diff"), reorder(text), restate(text), exponent(text), commute(text), flip(text), eqswap(text),
            delimiter(text), paren(text)]
    out, seen = [], {text}
    for p in made:
        if p is None or p.text in seen:
            continue
        v = Perturbation(p.name, p.text, p.check, p.meta, verify(text, p))
        if v.valid:
            seen.add(p.text)
            out.append(v)
    return out


SURFACE = {"delimiter", "paren"}  # notation only: no operand, name or order changes


def family(name: str) -> str:
    """rename_a and rename_b (const_sum and const_diff) are near-duplicates: one family."""
    return name.split("_")[0]


def one_per_family(ps: list) -> list:
    """First variant of each distinct family, in perturbations() order."""
    seen, out = set(), []
    for p in ps:
        if family(p.name) not in seen:
            seen.add(family(p.name))
            out.append(p)
    return out


def coverage(problems: list, minimum: int = 3) -> dict:
    """A problem is covered when it has `minimum` valid variants from distinct families."""
    rows = []
    for t in problems:
        ps = perturbations(t)
        rows.append({"variants": len(ps), "families": len({family(p.name) for p in ps}),
                     "semantic_families": len({family(p.name) for p in ps} - SURFACE),
                     "families_unflagged": len({family(p.name) for p in ps if p.check != "audit"}),
                     "names": [p.name for p in ps]})
    n = len(rows)
    return {"problems": n, "min_families": minimum,
            "with_min_variants": sum(r["variants"] >= minimum for r in rows),
            "with_min_families": sum(r["families"] >= minimum for r in rows),
            "with_min_semantic_families": sum(r["semantic_families"] >= minimum for r in rows),
            "with_min_families_unflagged": sum(r["families_unflagged"] >= minimum for r in rows),
            "covered_fraction": round(sum(r["families"] >= minimum for r in rows) / max(n, 1), 4), "rows": rows}


def main():
    import json
    from pathlib import Path

    import pandas as pd

    here = Path(__file__).resolve().parent.parent
    d = pd.read_parquet(here / "data/train-main-v2/train-00000-of-00001.parquet")
    small = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
             "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
    lab = d[d.model_id.isin(small) & d.is_robust.notna()]
    probs = list(dict.fromkeys(lab.problem))
    cov = coverage(probs)
    cases = sum(int((lab.problem == t).sum()) for t, r in zip(probs, cov["rows"]) if r["families"] >= 3)
    cov["labelled_cases"] = len(lab)
    cov["labelled_cases_with_min_families"] = cases
    print(json.dumps({k: v for k, v in cov.items() if k != "rows"}, indent=1))
    for i, r in enumerate(cov["rows"]):
        print(i, r["variants"], r["families"], r["names"])
    (here / "runs/invariance2").mkdir(parents=True, exist_ok=True)
    (here / "runs/invariance2/coverage.json").write_text(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
