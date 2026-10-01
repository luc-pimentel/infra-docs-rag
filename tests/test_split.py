import re

from infra_docs_rag.chunk.split import PARAGRAPH, SENTENCE, WORD, Tokens, fit, pack, split, trim

TEXT = (
    "Alpha beta gamma.\n\n"
    "Delta epsilon. Zeta eta theta.\nIota kappa.\n\n"
    "Lambda mu nu xi omicron pi rho sigma tau upsilon."
)


def words(text: str) -> Tokens:
    """One token per run of non-space characters, like the fake embedder."""
    return Tokens([m.span() for m in re.finditer(r"\S+", text)])


def texts(text: str, spans: list[tuple[int, int]]) -> list[str]:
    return [text[start:end] for start, end in spans]


def test_split_tiles_the_span_and_keeps_each_separator_with_the_text_before_it():
    units = split(TEXT, (0, len(TEXT)), PARAGRAPH)
    assert "".join(texts(TEXT, units)) == TEXT
    assert texts(TEXT, units)[0] == "Alpha beta gamma.\n\n"
    assert len(units) == 3


def test_sentences_end_before_a_capital_not_inside_an_abbreviation_or_a_version():
    text = "Use e.g. kubectl v1.2 here. Then run it! Next: done. `kubectl` works."
    assert texts(text, trim(text, split(text, (0, len(text)), SENTENCE))) == [
        "Use e.g. kubectl v1.2 here.",
        "Then run it!",
        "Next: done.",
        "`kubectl` works.",
    ]


def test_token_counts_add_up_across_a_cut():
    tokens, cut = words(TEXT), TEXT.index("Delta")
    assert (
        tokens.count(0, cut) + tokens.count(cut, len(TEXT)) == tokens.count(0, len(TEXT)) == len(TEXT.split())
    )


def test_pack_fills_each_span_to_the_budget_and_repeats_the_overlap():
    text = "a b c d e f g"
    units = split(text, (0, len(text)), WORD)
    assert texts(text, trim(text, pack(units, words(text), budget=3))) == ["a b c", "d e f", "g"]
    assert texts(text, trim(text, pack(units, words(text), budget=3, overlap=1))) == [
        "a b c",
        "c d e",
        "e f g",
    ]


def test_fit_cuts_at_the_coarsest_boundary_that_works():
    tokens = words(TEXT)
    # Paragraphs first: the first two together would be 10 tokens; the third alone is too big, and
    # with no line or sentence break inside it falls back to words.
    assert texts(TEXT, trim(TEXT, fit(TEXT, (0, len(TEXT)), tokens, budget=8))) == [
        "Alpha beta gamma.",
        "Delta epsilon. Zeta eta theta.\nIota kappa.",
        "Lambda mu nu xi omicron pi rho sigma",
        "tau upsilon.",
    ]
    # At 5 the middle paragraph no longer fits, so it breaks at its line break, not mid-sentence
    assert texts(TEXT, trim(TEXT, fit(TEXT, (0, len(TEXT)), tokens, budget=5)))[1:3] == [
        "Delta epsilon. Zeta eta theta.",
        "Iota kappa.",
    ]


def test_every_piece_fits_the_budget_and_pieces_overlap_only_inside_a_paragraph():
    tokens = words(TEXT)
    pieces = trim(TEXT, fit(TEXT, (0, len(TEXT)), tokens, budget=5, overlap=2))
    assert all(tokens.count(*piece) <= 5 for piece in pieces)
    assert texts(TEXT, pieces)[-3:] == [
        "Lambda mu nu xi omicron",
        "xi omicron pi rho sigma",
        "rho sigma tau upsilon.",
    ]
    assert texts(TEXT, pieces)[:3] == ["Alpha beta gamma.", "Delta epsilon. Zeta eta theta.", "Iota kappa."]


def test_text_with_no_whitespace_is_cut_at_token_edges():
    text = "abcdefghij"
    characters = Tokens([(i, i + 1) for i in range(len(text))])
    assert texts(text, fit(text, (0, len(text)), characters, budget=4)) == ["abcd", "efgh", "ij"]


def test_trim_drops_whitespace_and_empty_spans():
    text = "  a b \n\n   "
    assert texts(text, trim(text, [(0, 7), (7, len(text))])) == ["a b"]
