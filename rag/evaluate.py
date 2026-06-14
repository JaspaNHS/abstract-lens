"""
Systematic evaluation harness for the RAG synthesis layer.

Runs a fixed set of test questions and checks each answer programmatically for
several error classes — so you find regressions by running this, not by eyeballing
one query at a time.

Checks per answer:
  1. citations_valid   — no hallucinated [n] (numbers outside the retrieved range).
                         reconcile_citations() strips these; we assert it found none.
  2. citations_present — substantive (non-vague) answers carry >=1 citation.
  3. english           — answer is in English (Spanish-marker heuristic).
  4. vague_handled     — vague queries trigger a clarification, not a fabricated answer.
  5. grounded          — LLM-judge: every cited claim is actually supported by the
                         fragment it cites (faithfulness). Optional, costs tokens.

Usage:
  python evaluate.py            # rule-based checks only (fast, cheap)
  python evaluate.py --judge    # also run the LLM groundedness judge (slower)
"""

import os
import re
import sys
import json
import argparse
import anthropic

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import synthesize as S

# ── Test set ──────────────────────────────────────────────────────────────────
# kind: "narrow" | "broad" | "vague" | "out_of_scope"
TEST_QUESTIONS = [
    {"q": "What efficacy did teclistamab show in relapsed/refractory multiple myeloma?", "kind": "narrow"},
    {"q": "Predictors of neurological toxicity in patients receiving cilta-cel", "kind": "narrow"},
    {"q": "Luspatercept results in transfusion-dependent thalassemia", "kind": "narrow"},
    {"q": "What does the evidence say about CAR-T cell therapies across hematologic malignancies?", "kind": "broad"},
    {"q": "Overview of treatment approaches and outcomes in acute myeloid leukemia", "kind": "broad"},
    {"q": "cancer", "kind": "vague"},
    {"q": "treatment", "kind": "vague"},
    {"q": "What is the capital of France?", "kind": "out_of_scope"},
]

# Spanish markers that should NOT appear if the answer is English
SPANISH_MARKERS = re.compile(
    r"\b(los|las|para|según|fragmentos|pacientes|tratamiento|estudio|"
    r"podría|recuperados|sobre|también|enfermedad|riesgo|datos|"
    r"dosis|respuesta|no contienen)\b",
    re.IGNORECASE,
)

# Phrases that indicate a clarification / no-answer response
CLARIFY_MARKERS = re.compile(
    r"\b(too (vague|broad)|narrow (it )?down|could you (specify|clarify)|"
    r"which (disease|drug|population|outcome)|please specify|more specific|"
    r"do not contain enough information)\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM = """You are a strict fact-checker. You are given an ANSWER containing \
citation markers [n], and the FRAGMENTS those markers refer to. For each cited claim, \
decide whether the cited fragment(s) actually support it. Respond ONLY with JSON: \
{"supported": <int>, "unsupported": <int>, "notes": "<one short sentence>"}. \
Count each sentence-level factual claim once. A claim is 'supported' only if the cited \
fragment plainly contains that information; otherwise 'unsupported'."""


def check_rules(item, answer, sources, invalid):
    """Rule-based checks. Returns a dict of check_name -> (passed, detail)."""
    kind = item["kind"]
    results = {}

    # 1. No hallucinated citations
    results["citations_valid"] = (
        len(invalid) == 0,
        "ok" if not invalid else f"hallucinated {invalid}",
    )

    # 2. Citations present (only required for substantive answers)
    has_cite = bool(re.search(r"\[\d+\]", answer))
    if kind in ("narrow", "broad"):
        results["citations_present"] = (has_cite and len(sources) > 0,
                                        f"{len(sources)} cited")
    else:
        results["citations_present"] = (True, "n/a")

    # 3. English
    spanish_hits = SPANISH_MARKERS.findall(answer)
    results["english"] = (len(spanish_hits) == 0,
                          "ok" if not spanish_hits else f"spanish: {spanish_hits[:5]}")

    # 4. Vague / out-of-scope handled with a clarification or no-answer.
    #    Asking the user to narrow down is the correct behavior — it may still cite
    #    example abstracts while doing so, which is fine. What we want to rule out is a
    #    confident, definitive answer that ignores the ambiguity. We approximate "did
    #    not just answer confidently" by requiring an explicit clarification phrase.
    clarified = bool(CLARIFY_MARKERS.search(answer))
    if kind in ("vague", "out_of_scope"):
        results["vague_handled"] = (clarified,
                                    "clarified" if clarified else "answered without clarifying")
    else:
        results["vague_handled"] = (True, "n/a")

    # 5. Length calibration (soft): broad answers should be longer than narrow
    word_count = len(answer.split())
    results["_words"] = (True, str(word_count))

    return results


def judge_grounded(client, item, answer, sources):
    """LLM-judge groundedness. Returns (passed, detail)."""
    if not sources:
        return (True, "no citations to judge")
    frag_lines = []
    for s in sources:
        # Judge sees the SAME (full) fragment text the model was given, so an
        # "unsupported" verdict reflects the model, not a shorter judge window.
        frag_lines.append(f"[{s['num']}] {s['title']}: {s['text']}")
    frags = "\n\n".join(frag_lines)
    msg = f"ANSWER:\n{answer}\n\n{'='*50}\nFRAGMENTS:\n{frags}"
    try:
        resp = client.messages.create(
            model=S.MODEL,
            max_tokens=400,
            output_config={"effort": "low"},
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": msg}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        sup, unsup = data.get("supported", 0), data.get("unsupported", 0)
        total = sup + unsup
        rate = sup / total if total else 1.0
        # Pass if faithfulness rate >= 0.9 (tolerates one borderline judge call on
        # long answers). Below that signals a real groundedness problem.
        passed = rate >= 0.9
        detail = f"{sup}/{total} supported ({rate:.0%}) — {data.get('notes','')}"
        return (passed, detail)
    except Exception as e:
        return (None, f"judge error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="run LLM groundedness judge")
    parser.add_argument("--mode", default="with_figs", choices=["with_figs", "no_figs"])
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY")

    client = anthropic.Anthropic()
    cols = S.load_collections()

    print("=" * 74)
    print(f"  RAG EVALUATION  (mode={args.mode}, judge={args.judge})")
    print("=" * 74)

    rule_checks = ["citations_valid", "citations_present", "english", "vague_handled"]
    totals = {c: [0, 0] for c in rule_checks}      # [passed, total]
    judge_pass = judge_total = 0

    for item in TEST_QUESTIONS:
        q, kind = item["q"], item["kind"]
        answer, sources, invalid = S.synthesize(q, args.mode, client, cols)
        rules = check_rules(item, answer, sources, invalid)

        print(f"\n[{kind:12}] {q[:58]}")
        print(f"    words={rules['_words'][1]}  cited_sources={len(sources)}")
        for c in rule_checks:
            passed, detail = rules[c]
            totals[c][1] += 1
            if passed:
                totals[c][0] += 1
            mark = "PASS" if passed else "FAIL"
            print(f"    [{mark}] {c:18} {detail}")

        if args.judge and kind in ("narrow", "broad") and sources:
            gp, gd = judge_grounded(client, item, answer, sources)
            judge_total += 1
            if gp:
                judge_pass += 1
            mark = "PASS" if gp else ("WARN" if gp is None else "FAIL")
            print(f"    [{mark}] grounded           {gd}")

    print("\n" + "=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    for c in rule_checks:
        p, t = totals[c]
        print(f"  {c:20} {p}/{t}")
    if args.judge:
        print(f"  {'grounded':20} {judge_pass}/{judge_total}")
    print("=" * 74)


if __name__ == "__main__":
    main()
