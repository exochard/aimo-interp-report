"""Validity of the perturbation family on fixtures (no data directory needed).

  uv run --no-project --with pytest --with sympy python -m pytest test_invariance.py -q
"""

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sub"))
import invariance as I  # noqa: E402

GEOM = "Triangle $ABC$ has $AB = 20$, $AC = 13$, and $BC = 7$. Let $n$ be the area of $ABC$ and $m = 2$. Find $n + 3m$."
SYSTEM = ("Let $k$ be real such that the system\n\\begin{align*} |25-z|&=5\\\\ |z-4-k|&=|z-3i-k| \\\\ \\end{align*}\n"
          "has one solution. Find $k$, where $p$ and $q$ are relatively prime.")
RATIO = "The perimeters are in the ratio $125:6,$ and the slope is $y = \\frac{1}{2}x + \\frac{1}{3}$."
TIKZ = "Find $x$ if $x^2 = 49$.\n\\begin{tikzpicture}\\coordinate (A) at ($(D)!2!(E)$);\\end{tikzpicture}"
PROSE = "Alice has 5 sweets and Bob has 7."


def names(text):
    return {p.name for p in I.perturbations(text)}


def test_every_returned_variant_verifies_and_differs():
    for t in (GEOM, SYSTEM, RATIO, TIKZ):
        for p in I.perturbations(t):
            assert p.valid and p.text != t and I.verify(t, replace(p, valid=False))


def test_rename_roundtrips_and_picks_unused_letter():
    p = next(p for p in I.perturbations(GEOM) if p.name == "rename_a")
    old, new = p.meta["from"], p.meta["to"]
    assert I._free_letter(GEOM, new) and I._rename_text(p.text, new, old) == GEOM
    assert "\\frac" not in p.text and p.text.count("$") == GEOM.count("$")


def test_rename_leaves_commands_and_products_consistent():
    t = "Let $f(m)+f(n)=f(m+n+mn)$ with $\\mathbb{Z}_{\\geq 1}$ and $\\neq$ signs."
    out = I._rename_text(t, "n", "t")
    assert out == "Let $f(m)+f(t)=f(m+t+mt)$ with $\\mathbb{Z}_{\\geq 1}$ and $\\neq$ signs."


def test_rename_refuses_prose_letters_and_tikz():
    assert not any(p.name.startswith("rename") for p in I.perturbations("Find $a+b$ for distinct integers a and b."))
    assert not any(p.name.startswith("rename") for p in I.perturbations(TIKZ))


def test_constants_are_exact_and_skip_exponents_subscripts_and_tikz():
    t = "Find $17_b$ dividing $97_b$ with $10^{101}$, $x^2$ and $3\\pi$ and $41$."
    p = next(p for p in I.perturbations(t) if p.name == "const_sum")
    for _, _, old, new in p.meta["edits"]:
        assert old in {"3", "41"}
    t2 = I.perturbations(TIKZ)
    assert all("!2!" in p.text for p in t2)  # tikz block untouched


def test_constants_detect_a_corrupted_edit():
    p = next(p for p in I.perturbations(GEOM) if p.name == "const_sum")
    a, b, old, new = p.meta["edits"][0]
    bad = replace(p, text=p.text.replace(new, "(1+1)", 1))
    assert not I.verify(GEOM, bad)


def test_reorder_assignments_exchanged_and_checked():
    p = next(p for p in I.perturbations(GEOM) if p.name == "reorder")
    assert "$BC = 7$" in p.text.split(",")[0] or "BC = 7" in p.text[:40]
    assert p.check == "sympy"
    bad = replace(p, text=GEOM.replace("AB = 20", "AB = 21"))
    assert not I.verify(GEOM, bad)


def test_reorder_align_rows_and_pair_are_flagged_for_audit():
    p = next(p for p in I.perturbations(SYSTEM) if p.name == "reorder")
    assert p.check == "audit" and p.text.index("|z-4-k|") < p.text.index("|25-z|")
    assert "$q$ and $p$ are relatively prime" in p.text


def test_restate_ratio_and_decimal():
    p = next(p for p in I.perturbations(RATIO) if p.name == "restate")
    assert "\\frac{125}{6}" in p.text and "0.5x" in p.text and "\\frac{1}{3}" in p.text
    assert not I.verify(RATIO, replace(p, text=p.text.replace("0.5", "0.6")))


def test_prose_only_and_tikz_problems_get_no_unsafe_edit():
    assert I.perturbations(PROSE) == []
    for p in I.perturbations(TIKZ):
        assert p.text.endswith("\\end{tikzpicture}") and "($(D)!2!(E)$)" in p.text


def test_deterministic():
    assert I.perturbations(GEOM) == I.perturbations(GEOM)


def test_coverage_counts():
    c = I.coverage([GEOM, PROSE])
    assert c["problems"] == 2 and c["with_min_variants"] <= 1


# ---- round 2: families, article rule, context safety ------------------------------------

def get(text, name):
    return next((p for p in I.perturbations(text) if p.name == name), None)


def test_article_a_does_not_block_a_variable_a():
    t = "Find a pair with $a+b=7$ and a positive $c$."
    p = get(t, "rename_a") or get(t, "rename_b")
    assert p is not None and p.meta["from"] in "abc"
    assert " a pair" in p.text and " a positive" in p.text  # prose articles untouched


def test_prose_variable_a_still_blocks_rename():
    t = "Count integers a and b with $a+b+ab \\leq 100$."
    assert get(t, "rename_a") is None


def test_rename_capital_scalar_not_point_letters():
    t = "Let $N$ be the area of $ABC$ with $N=5$."
    p = get(t, "rename_a")
    assert p is not None and p.meta["from"] == "N" and "$ABC$" in p.text and "$N$" not in p.text
    assert I.verify(t, p)


def test_rename_allows_backslash_log():
    assert get("Solve $\\log_{2}x = 3$ for $x$.", "rename_a") is not None
    assert get("Solve $log x = 3$ for $x$.", "rename_a") is None


def test_exponent_sum_is_exact_and_detects_tamper():
    t = "Compute $10^{101}$ and $x^2$."
    p = get(t, "exponent")
    assert p and all(I._same(o, n) for _, _, o, n in p.meta["edits"])
    assert not I.verify(t, replace(p, text=p.text.replace("^{", "^{9+", 1)))


def test_commute_only_where_no_operator_binds():
    assert get("Find $m+n$.", "commute").text == "Find $n+m$."
    for t in ("Find $2 * b + c$.", "Find $-a+b$.", "Find $a+b^2$.", "Find $a+b(c)$.", "Find $x/a+b$."):
        assert get(t, "commute") is None, t
    assert get("Let $f(m + n + mn)$.", "commute").text == "Let $f(n + m + mn)$."


def test_flip_comparison():
    p = get("integer bases $b>9$ for which", "flip")
    assert p.text == "integer bases $9<b$ for which"
    assert get("with $a<b<c$", "flip") is None
    assert not I.verify("bases $b>9$", replace(p, text="bases $9>b$"))


def test_eqswap_pure_equalities_only():
    assert get("Triangle has $AB = BC.$", "eqswap").text == "Triangle has $BC = AB.$"
    for t in ("Take $x \\leq 3$.", "Let $f(1)=2, f(2)=3$.", "Sum $\\sum_{i=1}^n i$."):
        assert get(t, "eqswap") is None, t
    p = get("Triangle has $AB = BC.$", "eqswap")
    assert not I.verify("Triangle has $AB = BC.$", replace(p, text="Triangle has $BC = AC.$"))


def test_surface_notation_roundtrips_and_skips_tikz_and_intervals():
    t = "Let $f(n)=(n+1)^2$ on $[0,1)$ and $g(x)=(x)$."
    d = get(t, "delimiter")
    assert d.text.count("\\(") == 3 and I._undo_surface(d.text, "delimiter") == t
    pa = get(t, "paren")
    assert pa and "[0,1)" in pa.text and "\\left(n\\right)" in pa.text
    assert get(TIKZ, "delimiter").text.endswith("\\end{tikzpicture}")
    assert "($(D)!2!(E)$)" in get(TIKZ, "delimiter").text
    assert get("Use $\\left(x\\right)$.", "paren") is None


def test_reorder_with_trailing_comma_and_both_pair():
    t = "Suppose that $AB = 1628, AC = 932,$ and include both $24$ and $34$ as terms."
    p = get(t, "reorder")
    assert p and "AC = 932, AB = 1628" in p.text and "both $34$ and $24$" in p.text


def test_coverage_counts_families_not_variants():
    t = "Let $a$ be real with $a^2=4$ and $b=9$, $c=2$."
    ps = I.perturbations(t)
    assert len(ps) > len({I.family(p.name) for p in ps})  # rename_a/rename_b share a family
    assert len(I.one_per_family(ps)) == len({I.family(p.name) for p in ps})
    c = I.coverage([t, PROSE])
    assert c["problems"] == 2 and c["with_min_families"] <= 1
    assert c["with_min_semantic_families"] <= c["with_min_families"]


def test_every_family_variant_verifies_from_original_alone():
    for t in (GEOM, SYSTEM, RATIO, TIKZ, "Let $f(n)=(n+1)^2$, $b>9$, $m+n=3$ and $N=5$."):
        for p in I.perturbations(t):
            assert I.verify(t, replace(p, valid=False)), (p.name, t)
