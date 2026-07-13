"""
Task 5: Auto Tagging Support Tickets Using LLM
=================================================
Tags free-text support tickets with the top-3 most probable category tags
using an LLM, comparing zero-shot vs few-shot prompting.

Supports three LLM backends via the LLM_BACKEND env var: anthropic (default),
groq, openai.

Usage:
    export LLM_BACKEND="groq"
    export GROQ_API_KEY="gsk_..."
    python tagging.py                          # batch-tag data/support_tickets.csv
    python tagging.py --single "ticket text"    # tag one ad-hoc ticket
"""
import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "support_tickets.csv"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_NAMES = {
    "anthropic": "claude-sonnet-4-6",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}

TAG_TAXONOMY = [
    "Billing", "Refund", "Cancellation", "Login Issue", "Account",
    "Bug", "Performance", "Outage", "API", "Mobile App", "UI",
    "Feature Request", "How-To", "Team Management", "Security",
]

FEW_SHOT_EXAMPLES = [
    {
        "text": "I keep getting logged out every few minutes, even after checking 'remember me'.",
        "tags": ["Login Issue", "Bug", "Account"],
    },
    {
        "text": "Please add support for exporting reports as PDF, not just CSV.",
        "tags": ["Feature Request", "How-To"],
    },
    {
        "text": "I was billed $49 instead of the $29 plan I signed up for.",
        "tags": ["Billing", "Account", "Refund"],
    },
]


def _get_backend() -> str:
    return os.environ.get("LLM_BACKEND", "anthropic")


def _client():
    """Returns (backend_name, client) based on the LLM_BACKEND env var."""
    backend = _get_backend()

    if backend == "groq":
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set. Run `$env:GROQ_API_KEY=\"gsk_...\"` first.")
        return backend, Groq(api_key=api_key)

    elif backend == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Run `$env:OPENAI_API_KEY=\"sk-...\"` first.")
        return backend, OpenAI(api_key=api_key)

    else:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set. Run `$env:ANTHROPIC_API_KEY=\"sk-ant-...\"` first.")
        return backend, Anthropic(api_key=api_key)


def _build_prompt(ticket_text: str, few_shot: bool) -> str:
    taxonomy_str = ", ".join(TAG_TAXONOMY)

    examples_block = ""
    if few_shot:
        lines = []
        for ex in FEW_SHOT_EXAMPLES:
            lines.append(f'Ticket: "{ex["text"]}"\nTags: {json.dumps(ex["tags"])}')
        examples_block = "\n\nHere are some labeled examples:\n" + "\n\n".join(lines) + "\n"

    prompt = f"""You are a support-ticket tagging assistant.

Choose the top 3 most probable tags for the ticket below, ranked from most to
least relevant, using ONLY tags from this fixed taxonomy:
{taxonomy_str}
{examples_block}
Now tag this ticket:
Ticket: "{ticket_text}"

Respond with ONLY a JSON object of the form {{"tags": ["tag1", "tag2", "tag3"]}}
and nothing else -- no explanation, no markdown formatting."""
    return prompt


def _parse_tags(raw_response: str) -> list:
    """Extracts and validates the {"tags": [...]} JSON from the model response."""
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    tags = parsed.get("tags", [])
    # keep only tags that are in the taxonomy (case-insensitive match)
    taxonomy_lower = {t.lower(): t for t in TAG_TAXONOMY}
    validated = [taxonomy_lower[t.lower()] for t in tags if t.lower() in taxonomy_lower]
    return validated[:3]


def tag_ticket(backend: str, client, ticket_text: str, few_shot: bool = False) -> list:
    prompt = _build_prompt(ticket_text, few_shot=few_shot)
    model_name = MODEL_NAMES[backend]

    if backend == "anthropic":
        response = client.messages.create(
            model=model_name,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(block.text for block in response.content if block.type == "text")
    else:
        # groq and openai both expose an OpenAI-compatible chat.completions API
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.choices[0].message.content

    return _parse_tags(raw_text)


def batch_tag():
    backend, client = _client()
    print(f"Using LLM backend: {backend} ({MODEL_NAMES[backend]})")
    df = pd.read_csv(DATA_PATH)

    zero_shot_preds, few_shot_preds = [], []
    for _, row in df.iterrows():
        print(f"Tagging ticket {row['ticket_id']}...")
        zero_shot_preds.append(";".join(tag_ticket(backend, client, row["text"], few_shot=False)))
        few_shot_preds.append(";".join(tag_ticket(backend, client, row["text"], few_shot=True)))

    df["zero_shot_tags"] = zero_shot_preds
    df["few_shot_tags"] = few_shot_preds

    out_path = MODELS_DIR / "predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")
    print(df[["ticket_id", "true_tags", "zero_shot_tags", "few_shot_tags"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, default=None, help="Tag a single ad-hoc ticket string")
    args = parser.parse_args()

    if args.single:
        backend, client = _client()
        zero_shot = tag_ticket(backend, client, args.single, few_shot=False)
        few_shot = tag_ticket(backend, client, args.single, few_shot=True)
        print(f"Ticket: {args.single}")
        print(f"Zero-shot tags: {zero_shot}")
        print(f"Few-shot tags:  {few_shot}")
    else:
        batch_tag()


if __name__ == "__main__":
    main()
