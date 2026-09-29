from infra_docs_rag.ingest.dedup import mark_duplicates

PROBES = (
    "The kubelet uses liveness probes to know when to restart a container. Readiness probes decide "
    "when a container can start accepting traffic, and startup probes hold the other two back until "
    "a slow application has finished booting."
)


def test_exact_copy_with_different_whitespace_is_flagged(make_doc):
    docs = [make_doc("original", PROBES), make_doc("copy", PROBES.replace(" ", "  ").upper())]
    mark_duplicates(docs)
    assert docs[0].duplicate is None
    assert docs[1].duplicate.kind == "exact" and docs[1].duplicate.of == "doc_original"


def test_near_copy_is_flagged_and_unrelated_page_is_kept(make_doc):
    edited = PROBES.replace("slow application", "slow Java application")
    unrelated = "Services expose a set of Pods behind one stable virtual IP address and DNS name."
    docs = [make_doc("stable", PROBES), make_doc("latest", edited), make_doc("service", unrelated)]
    pairs = mark_duplicates(docs)
    assert docs[1].duplicate.kind == "near" and docs[1].duplicate.of == "doc_stable"
    assert docs[2].duplicate is None
    assert len(pairs) == 2  # "latest" is compared with "stable"; "service" with "stable" only
