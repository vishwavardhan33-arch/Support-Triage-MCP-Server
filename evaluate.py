"""
Evaluation harness for the support-triage MCP server.

Measures two different things:

1. Routing accuracy — how often classify_ticket's predicted category and
   priority match the mock dataset's ground-truth `category`/`priority`
   fields (generate_data.py bakes these in as labels).

2. LLM-as-judge — there's no ground truth for the *quality* of the drafted
   suggested_response text, so a second LLM call scores it on relevance,
   professionalism, and actionability. Using a different/larger model as
   judge than the one doing the classifying reduces self-judging bias
   (a model tends to rate its own outputs generously).

Both steps run entirely on local Ollama models — no external API, no cost.

Usage:
    python evaluate.py --limit 10
    python evaluate.py --limit 10 --judge-model mistral
    python evaluate.py --limit 10 --no-judge     # routing accuracy only, faster
"""

import argparse
import json
import statistics
from pathlib import Path

import server

JUDGE_SYSTEM_PROMPT = """You are evaluating the quality of an AI-drafted first response \
to a support ticket at a home loan partner-support desk. Score the response honestly \
and critically - do not default to high scores.

Respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:
{
  "relevance": <integer 1-5, does it address what the ticket is actually about>,
  "professionalism": <integer 1-5, tone and clarity>,
  "actionability": <integer 1-5, does it tell the sender what happens next>,
  "overall": <integer 1-5>,
  "justification": "<one sentence explaining the scores>"
}"""


def _build_judge_prompt(ticket: dict, suggested_response: str) -> str:
    return (
        f"TICKET SUBJECT: {ticket['subject']}\n"
        f"TICKET BODY: {ticket['body']}\n\n"
        f"AI-DRAFTED RESPONSE TO EVALUATE:\n{suggested_response}"
    )


def judge_response(ticket: dict, suggested_response: str, judge_model: str) -> dict:
    raw = server._call_ollama(
        JUDGE_SYSTEM_PROMPT,
        _build_judge_prompt(ticket, suggested_response),
        model=judge_model,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"error": "judge output not valid JSON", "raw_output": raw}


def run_eval(limit: int, judge_model: str, run_judge: bool) -> list[dict]:
    tickets = server._load_tickets()[:limit]
    classify_fn = getattr(server.classify_ticket, "fn", server.classify_ticket)
    results = []

    for i, ticket in enumerate(tickets, 1):
        print(f"[{i}/{len(tickets)}] Classifying {ticket['id']}...")
        predicted = classify_fn(ticket["id"])

        if "error" in predicted:
            print(f"  -> error: {predicted['error']}")
            results.append({"ticket_id": ticket["id"], "error": predicted["error"]})
            continue

        pred_category = (predicted.get("category") or "").strip().lower()
        pred_priority = (predicted.get("priority") or "").strip().lower()

        record = {
            "ticket_id": ticket["id"],
            "true_category": ticket["category"],
            "predicted_category": predicted.get("category"),
            "category_correct": pred_category == ticket["category"].strip().lower(),
            "true_priority": ticket["priority"],
            "predicted_priority": predicted.get("priority"),
            "priority_correct": pred_priority == ticket["priority"].strip().lower(),
            "suggested_response": predicted.get("suggested_response"),
        }

        if run_judge:
            print(f"  -> judging response quality with '{judge_model}'...")
            record["judge"] = judge_response(ticket, predicted.get("suggested_response", ""), judge_model)

        results.append(record)

    return results


def summarize(results: list[dict]) -> None:
    scored = [r for r in results if "error" not in r]
    errored = [r for r in results if "error" in r]

    if errored:
        print(f"\n{len(errored)} ticket(s) failed to classify (see eval_results.json for details).")

    if not scored:
        print("No successful classifications to summarize.")
        return

    category_correct = sum(r["category_correct"] for r in scored)
    priority_correct = sum(r["priority_correct"] for r in scored)

    print("\n=== Routing accuracy ===")
    print(f"Category accuracy: {category_correct / len(scored):.1%} ({category_correct}/{len(scored)})")
    print(f"Priority accuracy: {priority_correct / len(scored):.1%} ({priority_correct}/{len(scored)})")

    misrouted = [r for r in scored if not r["category_correct"]]
    if misrouted:
        print("\nMisrouted tickets:")
        for r in misrouted:
            print(f"  {r['ticket_id']}: true='{r['true_category']}' predicted='{r['predicted_category']}'")

    judged = [r["judge"] for r in scored if "judge" in r and "overall" in r["judge"]]
    if judged:
        print("\n=== LLM-as-judge: suggested_response quality (1-5 scale) ===")
        for key in ["relevance", "professionalism", "actionability", "overall"]:
            vals = [j[key] for j in judged if isinstance(j.get(key), (int, float))]
            if vals:
                print(f"{key.capitalize():<16} avg {statistics.mean(vals):.2f}  (n={len(vals)})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the support-triage classifier")
    parser.add_argument("--limit", type=int, default=10,
                         help="Number of tickets to evaluate (default: 10). "
                              "Each ticket costs 1-2 local LLM calls, so start small on CPU-only machines.")
    parser.add_argument("--judge-model", type=str, default=None,
                         help="Model to use as judge. Defaults to the same model as classify_ticket "
                              "(OLLAMA_MODEL env var), but a different/larger model reduces self-judging bias, "
                              "e.g. --judge-model mistral")
    parser.add_argument("--no-judge", action="store_true",
                         help="Skip the LLM-as-judge step and only report routing accuracy (faster).")
    args = parser.parse_args()

    judge_model = args.judge_model or server.OLLAMA_MODEL
    results = run_eval(args.limit, judge_model, run_judge=not args.no_judge)

    out_path = Path(__file__).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-ticket results written to {out_path}")

    summarize(results)


if __name__ == "__main__":
    main()
