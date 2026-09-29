from infra_docs_rag.embed.report import block, ordinal


def test_a_code_block_outlasts_the_backticks_inside_it():
    snippet = "   ``` kubectl create namespace argocd"  # a search hit that quotes a Markdown fence
    fenced = block(snippet)
    assert fenced.startswith("````text\n") and fenced.endswith("\n````\n")
    assert block("plain text").startswith("```text\n")


def test_ordinals():
    ranks = [1, 2, 3, 4, 11, 12, 13, 21, 102]
    assert [ordinal(n) for n in ranks] == [
        "1st",
        "2nd",
        "3rd",
        "4th",
        "11th",
        "12th",
        "13th",
        "21st",
        "102nd",
    ]
