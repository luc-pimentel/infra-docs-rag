from infra_docs_rag.ingest.parsers.base import assemble
from infra_docs_rag.ingest.parsers.markdown import parse_markdown

DOC = b"""---
title: ConfigMaps
weight: 20
---
<!-- overview -->
{{< glossary_definition term_id="configmap" length="all" >}}
A {{< glossary_tooltip text="Pod" term_id="pod" >}} reads its settings from a [ConfigMap](/docs/configmap/).

## Using ConfigMaps

```shell
# a shell comment, not a heading
kubectl get configmap <name>
```

## What's next
"""


def test_front_matter_is_the_title_and_template_syntax_is_removed():
    parsed = parse_markdown(DOC, "configmap.md")
    text, _, _ = assemble(parsed.parts)
    assert parsed.title == "ConfigMaps"
    assert "A Pod reads its settings from a ConfigMap." in text
    assert "{{" not in text and "<!--" not in text and "weight: 20" not in text


def test_code_blocks_keep_placeholders_and_comments():
    text, sections, _ = assemble(parse_markdown(DOC, "configmap.md").parts)
    assert "kubectl get configmap <name>" in text
    assert "# a shell comment, not a heading" in text
    assert [s.title for s in sections] == ["ConfigMaps", "Using ConfigMaps"]  # "What's next" is empty


def test_section_spans_point_at_their_headings():
    text, sections, _ = assemble(parse_markdown(DOC, "configmap.md").parts)
    for section in sections:
        assert text[section.start : section.end].startswith(section.title)
    assert sections[1].path == ["ConfigMaps", "Using ConfigMaps"]
