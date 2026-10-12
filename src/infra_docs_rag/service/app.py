"""The service: the released pipeline behind HTTP. `POST /answer` is `infra-docs-rag answer` with a JSON
body; `GET /health` is `pipeline health`; `GET /traces/{id}` reads the trace log; `GET /release` and
`GET /versions` say what is answering. Everything loads once at startup (the index, the models, the
prompt) under the current release, and a request never touches the pipeline's state: the service reads
an index version, it does not build one.

    uv run infra-docs-rag serve                      # http://127.0.0.1:8000, docs at /docs
    curl -s localhost:8000/answer -d '{"question": "How do I undo a bad release?"}' -H 'content-type: application/json'
"""

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..embed.embedders import load_embedder
from ..embed.index import Index
from ..evaluation.versions import Version, load_versions
from ..generate.answer import Answer, answer, marked
from ..generate.cache import CACHE, CachedGenerator, GenerationCache
from ..generate.generator import DEFAULT_EFFORT, Generator, load_generator
from ..generate.generator import DEFAULT_MODEL as DEFAULT_GENERATOR
from ..generate.judge import Judge, Judgement
from ..generate.prompt import ABSTAIN
from ..generate.prompts import Prompt, load_prompt
from ..observe.health import checks, worst
from ..observe.summary import format_summary, summarize
from ..observe.trace import TRACES, Trace, TraceLog, trace_answer
from ..pipeline.commands import index_for, lineage, resolve_version
from ..pipeline.manifest import PipelineManifest
from ..pipeline.releases import RELEASES, Releases
from ..pipeline.stages import Config, Paths, build_pipeline
from ..retrieve.rerank import load_reranker
from ..retrieve.retriever import Retriever

INDEX_DIR = Path("data/index")


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    judge: bool = False  # have the judge read the answer too (one more call)
    no_cache: bool = False


class CitationOut(BaseModel):
    n: int
    source_id: str
    citation: str
    uri: str
    cited_text: str
    span: str


class SourceOut(BaseModel):
    n: int
    chunk_id: str
    citation: str
    uri: str
    score: float
    cosine: float | None
    relevance: float | None


class JudgementOut(BaseModel):
    grounded: float
    fabricated: float
    gap_named: float
    model: str


class AnswerResponse(BaseModel):
    trace_id: str
    outcome: str
    answer: str  # the text with [n] after each cited span; the abstention sentence when there is none
    reason: str = ""
    citations: list[CitationOut] = []
    sources: list[SourceOut] = []
    cited_share: float = 0.0
    claim_share: float = 0.0
    version: str
    lineage: PipelineManifest
    stages_ms: dict[str, float] = {}
    total_ms: float = 0.0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    degraded: str | None = None
    judgement: JudgementOut | None = None


@dataclass
class Service:
    """Everything a request needs, loaded once."""

    version: Version
    index: Index
    index_id: str
    retriever: Retriever
    generator: Generator
    prompt: Prompt
    judge: Judge | None
    traces: TraceLog
    releases_path: Path = RELEASES
    index_dir: Path = INDEX_DIR

    @property
    def lineage(self) -> PipelineManifest:
        return lineage(
            self.version, self.index, self.index_id, self.prompt, self.generator.name, self.generator.model
        )

    def answer(
        self, question: str, judge: bool = False, no_cache: bool = False
    ) -> tuple[Answer, Judgement | None, Trace]:
        gen = self.generator
        if no_cache and isinstance(gen, CachedGenerator):
            gen = gen.inner
        a = answer(question, self.retriever, gen, self.version.retrieval, retries=1)
        reading = None
        if judge and self.judge is not None and a.outcome == "answered":
            try:
                reading = self.judge.judge(a.context, question, marked(a))
            except ValueError:
                reading = None
        t = trace_answer(
            a, self.version.name, self.lineage, self.prompt.version(gen.name), reading, gen.prices
        )
        self.traces.append(t)
        return a, reading, t

    def health(self):
        pipeline = build_pipeline(
            Paths(), Config(model=self.version.embedding), lambda: self.retriever.embedder
        )
        return checks(
            self.index_dir, self.version.embedding, Releases.load(self.releases_path), pipeline, self.traces
        )


def load_service(
    generator: str | None = None,
    effort: str = DEFAULT_EFFORT,
    version_name: str | None = None,
    use_cache: bool = True,
    with_judge: bool = True,
) -> Service:
    """The released pipeline, or `version_name`, loaded for serving. The generator comes from
    INFRA_DOCS_RAG_GENERATOR (default the API model) and the effort from INFRA_DOCS_RAG_EFFORT."""
    generator = generator or os.environ.get("INFRA_DOCS_RAG_GENERATOR", DEFAULT_GENERATOR)
    effort = os.environ.get("INFRA_DOCS_RAG_EFFORT", effort)
    version, pinned = resolve_version(version_name)
    index, index_id = index_for(version, pinned, INDEX_DIR)
    retrieval = version.retrieval
    reranker = load_reranker(retrieval.rerank.model) if retrieval.rerank is not None else None
    retriever = Retriever(index, load_embedder(version.embedding), reranker)
    prompt = load_prompt(version.prompt)
    gen: Generator = load_generator(generator, effort, version.prompt)
    if use_cache:
        gen = CachedGenerator(gen, GenerationCache(CACHE), prompt.version(gen.name))
    judge = None
    if with_judge:
        from ..generate.judge import load_judge

        try:
            judge = load_judge()
        except ValueError:
            judge = None
    return Service(version, index, index_id, retriever, gen, prompt, judge, TraceLog(TRACES))


def response_of(
    a: Answer, reading: Judgement | None, t: Trace, version: Version, lineage_: PipelineManifest
) -> AnswerResponse:
    g = a.generated
    text = marked(a) if a.outcome == "answered" else (a.text or ABSTAIN)
    if a.outcome in ("error", "refused", "cut-off"):
        text = ABSTAIN
    return AnswerResponse(
        trace_id=t.trace_id,
        outcome=a.outcome,
        answer=text,
        reason=a.reason,
        citations=[
            CitationOut(
                n=c.n,
                source_id=c.chunk.source_id,
                citation=c.chunk.citation(),
                uri=c.chunk.source_uri,
                cited_text=c.cited_text,
                span=c.span,
            )
            for c in a.citations
        ],
        sources=[
            SourceOut(
                n=s.n,
                chunk_id=s.chunk.chunk_id,
                citation=s.chunk.citation(),
                uri=s.chunk.source_uri,
                score=float(s.hit.score),
                cosine=next((h.cosine for h in t.context if h.rank == s.n), None),
                relevance=next((h.relevance for h in t.context if h.rank == s.n), None),
            )
            for s in a.context.sources
        ],
        cited_share=a.cited_share,
        claim_share=a.claim_share,
        version=version.name,
        lineage=lineage_,
        stages_ms=t.stages,
        total_ms=t.total_ms,
        model=g.model if g else "",
        input_tokens=g.usage.input_tokens if g else 0,
        output_tokens=g.usage.output_tokens if g else 0,
        cost_usd=t.cost_usd,
        cached=bool(g and g.cached),
        degraded=a.context.degraded,
        judgement=JudgementOut(
            grounded=reading.grounded,
            fabricated=reading.fabricated,
            gap_named=reading.gap_named,
            model=reading.model,
        )
        if reading
        else None,
    )


def create_app(service: Service | None = None, loader=load_service) -> FastAPI:
    """The app. A test passes a `service` built from fakes; the real one loads at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or loader()
        yield

    app = FastAPI(
        title="infra-docs-rag",
        description="Grounded answers about Kubernetes, Prometheus and Argo CD, each citing the documentation it came from.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def svc(request: Request) -> Service:
        return request.app.state.service

    @app.post("/answer", response_model=AnswerResponse)
    def post_answer(body: AnswerRequest, request: Request) -> AnswerResponse:
        s = svc(request)
        a, reading, t = s.answer(body.question, body.judge, body.no_cache)
        return response_of(a, reading, t, s.version, s.lineage)

    @app.get("/health")
    def get_health(request: Request):
        found = svc(request).health()
        overall = worst(found)
        body = {
            "status": overall,
            "checks": [{"name": c.name, "status": c.status, "detail": c.detail} for c in found],
        }
        return JSONResponse(body, status_code=503 if overall == "fail" else 200)

    @app.get("/release")
    def get_release(request: Request):
        s = svc(request)
        live = Releases.load(s.releases_path).live
        return {
            "serving": {
                "version": s.version.name,
                "index": f"{s.version.embedding}/{s.index_id}",
                "lineage": s.lineage.model_dump(),
            },
            "release": live.model_dump() if live else None,
        }

    @app.get("/versions")
    def get_versions():
        return {name: v.model_dump() for name, v in load_versions().items()}

    @app.get("/traces/summary")
    def get_summary(request: Request):
        s = svc(request)
        everything = s.traces.read()
        summary = summarize(everything, s.index_id)
        return {"text": format_summary(summary), "flags": summary.flags, "traces": summary.traces}

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str, request: Request):
        found = svc(request).traces.find(trace_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no trace {trace_id}")
        return found.model_dump()

    @app.get("/traces")
    def get_traces(request: Request, last: int = 10):
        return [t.model_dump() for t in svc(request).traces.read(last=max(1, min(last, 200)))]

    return app


app = create_app()
