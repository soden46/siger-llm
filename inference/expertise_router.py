from __future__ import annotations

import re
from dataclasses import dataclass

from inference.generator import Generator
from inference.lampung_pipeline import LampungPipeline


@dataclass(frozen=True)
class ExpertiseDraft:
    domain: str
    text: str
    source: str


@dataclass(frozen=True)
class ExpertiseResponse:
    text: str
    route: str
    source: str
    domains: list[str]
    task_summary: str
    drafts: list[ExpertiseDraft]


@dataclass(frozen=True)
class ExpertiseSpec:
    name: str
    keywords: tuple[str, ...]
    system_prompt: str
    focus_prompt: str


class ExpertiseOrchestrator:
    """General expertise router that decomposes a user task by domain.

    This is a lightweight runtime counterpart to the separated expertise
    training configs. It keeps domain logic outside the core model while giving
    the CLI/API a place to plan, call focused expertise prompts, and synthesize
    a final answer.
    """

    LAMPUNG_HINTS = {
        "nyak",
        "nikeu",
        "sekam",
        "mettei",
        "gham",
        "ikam",
        "tiyan",
        "punyeu",
        "mengan",
        "haga",
        "ago",
        "dak",
        "paghek",
        "nuwo",
        "wawai",
        "tabik",
    }

    EXPERTISE: tuple[ExpertiseSpec, ...] = (
        ExpertiseSpec(
            name="lampung",
            keywords=("lampung", "dialek o", "terjemah", "translate"),
            system_prompt=(
                "Kamu adalah expertise Bahasa Lampung Dialek O, Bahasa Indonesia, "
                "dan English. Fokus pada terjemahan, makna kata, dan struktur kalimat."
            ),
            focus_prompt="Analisis bagian Bahasa Lampung, terjemahan, makna kata, dan struktur kalimat.",
        ),
        ExpertiseSpec(
            name="indonesian",
            keywords=("bahasa indonesia", "tulis", "ringkas", "parafrase", "ejaan", "kalimat"),
            system_prompt=(
                "Kamu adalah expertise Bahasa Indonesia. Fokus pada kejelasan, "
                "struktur jawaban, gaya bahasa, dan akurasi makna."
            ),
            focus_prompt="Rapikan maksud tugas, bahasa, struktur, dan cara penyajian dalam Bahasa Indonesia.",
        ),
        ExpertiseSpec(
            name="programming_basic",
            keywords=(
                "python",
                "fungsi",
                "string",
                "list",
                "dictionary",
                "loop",
                "if else",
                "syntax",
                "sintaks",
            ),
            system_prompt=(
                "Kamu adalah expertise programming dasar. Fokus pada sintaks, "
                "fungsi kecil, struktur data dasar, dan penjelasan kode sederhana."
            ),
            focus_prompt="Pecah kebutuhan coding dasar, sintaks, fungsi, dan logika sederhana.",
        ),
        ExpertiseSpec(
            name="programming_intermediate",
            keywords=(
                "algoritma",
                "debug",
                "bug",
                "oop",
                "class",
                "rekursi",
                "sorting",
                "optimasi",
                "complexity",
            ),
            system_prompt=(
                "Kamu adalah expertise algoritma menengah dan debugging. Fokus pada "
                "alur state, kompleksitas, OOP, edge case, dan perbaikan bug."
            ),
            focus_prompt="Analisis algoritma, debugging, state, kompleksitas, dan edge case.",
        ),
        ExpertiseSpec(
            name="programming_expert",
            keywords=(
                "api",
                "fastapi",
                "laravel",
                "docker",
                "postgres",
                "security",
                "compliance",
                "logging",
                "production",
                "architecture",
            ),
            system_prompt=(
                "Kamu adalah expertise software engineering expert. Fokus pada "
                "arsitektur, API, database, security, testing, observability, dan deployment."
            ),
            focus_prompt="Rancang solusi software engineering yang production-ready dan aman.",
        ),
        ExpertiseSpec(
            name="reasoning",
            keywords=("hitung", "matematika", "logika", "analisis", "kenapa", "buktikan", "reasoning"),
            system_prompt=(
                "Kamu adalah expertise reasoning. Fokus pada pemecahan masalah, "
                "asumsi, langkah logis, dan batas keyakinan."
            ),
            focus_prompt="Uraikan problem menjadi asumsi, langkah penalaran, dan kesimpulan.",
        ),
        ExpertiseSpec(
            name="general_knowledge",
            keywords=("apa itu", "jelaskan", "sejarah", "konsep", "manfaat", "contoh"),
            system_prompt=(
                "Kamu adalah expertise pengetahuan umum. Fokus pada jawaban faktual, "
                "ringkas, dan mudah dipahami."
            ),
            focus_prompt="Jawab sisi pengetahuan umum dan definisi inti dari tugas.",
        ),
    )

    SYNTHESIS_SYSTEM_PROMPT = (
        "Kamu adalah general expertise SigerLM. Tugasmu merangkum permintaan user, "
        "membagi ke expertise domain yang relevan, menggabungkan draft jawaban, "
        "menghapus duplikasi, dan menyajikan jawaban final yang jelas."
    )

    def __init__(
        self,
        generator: Generator,
        lampung: LampungPipeline | None = None,
    ) -> None:
        self.generator = generator
        self.lampung = lampung

    def route(
        self,
        text: str,
        max_new_tokens: int = 160,
        max_domains: int = 4,
    ) -> ExpertiseResponse:
        task_summary = self.summarize_task(text)
        specs = self.detect_expertise(text, max_domains=max_domains)
        drafts = [self._run_expertise(spec, text, task_summary, max_new_tokens) for spec in specs]

        if len(drafts) == 1:
            final_text = drafts[0].text.strip()
            source = drafts[0].source
        else:
            final_text = self._synthesize(text, task_summary, drafts, max_new_tokens)
            source = "expertise synthesis"

        return ExpertiseResponse(
            text=final_text.strip(),
            route="expertise_orchestrator",
            source=source,
            domains=[draft.domain for draft in drafts],
            task_summary=task_summary,
            drafts=drafts,
        )

    def summarize_task(self, text: str) -> str:
        normalized = " ".join(text.strip().split())
        if len(normalized) <= 180:
            return normalized
        return normalized[:177].rstrip() + "..."

    def detect_expertise(self, text: str, *, max_domains: int = 4) -> list[ExpertiseSpec]:
        normalized = self._norm(text)
        scored: list[tuple[int, ExpertiseSpec]] = []

        for spec in self.EXPERTISE:
            score = sum(1 for keyword in spec.keywords if keyword in normalized)
            if spec.name == "lampung" and self.looks_lampung(normalized):
                score += 3
            if score:
                scored.append((score, spec))

        if not scored:
            scored.append((1, self._spec("general_knowledge")))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[ExpertiseSpec] = []
        seen: set[str] = set()
        for _, spec in scored:
            if spec.name in seen:
                continue
            selected.append(spec)
            seen.add(spec.name)
            if len(selected) >= max_domains:
                break
        return selected

    def looks_lampung(self, text: str) -> bool:
        words = set(re.findall(r"[a-z']+", text.lower()))
        return len(words & self.LAMPUNG_HINTS) >= 2

    def _run_expertise(
        self,
        spec: ExpertiseSpec,
        user_text: str,
        task_summary: str,
        max_new_tokens: int,
    ) -> ExpertiseDraft:
        if spec.name == "lampung" and self.lampung is not None and self.looks_lampung(self._norm(user_text)):
            response = self.lampung.reason_lo_to_id(user_text, max_new_tokens=max(80, max_new_tokens // 2))
            return ExpertiseDraft(spec.name, response.text, response.source)

        prompt = (
            f"<|system|>{spec.system_prompt}<|end_turn|>\n"
            f"<|user|>Ringkasan tugas: {task_summary}\n\n"
            f"Fokus expertise: {spec.focus_prompt}\n\n"
            f"Permintaan user:\n{user_text}<|end_turn|>\n"
            "<|assistant|>"
        )
        text = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_k=20,
            top_p=0.8,
            stop_tokens=[
                self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
                self.generator.tokenizer.eos_id,
            ],
        )
        return ExpertiseDraft(spec.name, text.strip(), "model expertise generation")

    def _synthesize(
        self,
        user_text: str,
        task_summary: str,
        drafts: list[ExpertiseDraft],
        max_new_tokens: int,
    ) -> str:
        draft_text = "\n\n".join(
            f"[{draft.domain}]\n{draft.text}" for draft in drafts if draft.text.strip()
        )
        prompt = (
            f"<|system|>{self.SYNTHESIS_SYSTEM_PROMPT}<|end_turn|>\n"
            f"<|user|>Ringkasan tugas: {task_summary}\n\n"
            f"Permintaan asli:\n{user_text}\n\n"
            f"Draft expertise:\n{draft_text}\n\n"
            "Gabungkan menjadi jawaban final yang praktis, ringkas, dan tidak berulang.<|end_turn|>\n"
            "<|assistant|>"
        )
        return self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_k=20,
            top_p=0.8,
            stop_tokens=[
                self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
                self.generator.tokenizer.eos_id,
            ],
        )

    def _spec(self, name: str) -> ExpertiseSpec:
        for spec in self.EXPERTISE:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def _norm(self, text: str) -> str:
        return " ".join(text.strip().lower().split())
