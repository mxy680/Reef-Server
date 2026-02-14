"""
Test: Does cancelling a Groq streaming request reduce token billing?

Approach:
  1. One full streaming request — record reported usage (~800-900 completion tokens)
  2. Five cancelled streaming requests — cancel after ~10 tokens each
  3. If cancellation saves tokens, the 5 cancelled requests should bill for
     ~50 completion tokens total. If it doesn't, they'd bill for ~4000-4500.
  4. Check Groq dashboard to compare actual billed usage.

Also uses httpx to capture raw response headers for any usage metadata.
"""

import os, time, sys, json, httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    sys.exit("GROQ_API_KEY not set — add it to .env")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
PROMPT = (
    "Write a detailed 500-word essay about the history of mathematics, "
    "covering ancient civilizations through modern times. "
    "Include specific dates, names, and contributions."
)


def test_full_streaming():
    """Streaming full completion — baseline for comparison."""
    print("=" * 60)
    print("BASELINE: Full streaming request")
    print("=" * 60)
    t0 = time.time()
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = 0
    text = ""
    usage = None
    for chunk in stream:
        chunks += 1
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
        if chunk.usage:
            usage = chunk.usage
    elapsed = time.time() - t0

    pt = usage.prompt_tokens if usage else "?"
    ct = usage.completion_tokens if usage else "?"
    print(f"  Time:              {elapsed:.2f}s")
    print(f"  Prompt tokens:     {pt}")
    print(f"  Completion tokens: {ct}")
    print(f"  Response length:   {len(text)} chars")
    print()
    return pt, ct


def test_cancelled_httpx(cancel_after_chunks=10):
    """Use httpx directly to cancel stream and capture headers."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.time()
    content_chunks = 0
    partial_text = ""
    response_headers = {}

    with httpx.Client(timeout=30) as http:
        with http.stream("POST", url, headers=headers, json=body) as resp:
            response_headers = dict(resp.headers)
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    content_chunks += 1
                    partial_text += choices[0]["delta"]["content"]
                # Check for usage in chunk
                if "usage" in chunk and chunk["usage"]:
                    print(f"    (usage in chunk: {chunk['usage']})")
                if content_chunks >= cancel_after_chunks:
                    break  # exits the stream context manager, closing connection

    elapsed = time.time() - t0
    return elapsed, content_chunks, partial_text, response_headers


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Groq Cancellation Billing Test")
    print("=" * 60)
    print("\nStep 1: Note your current usage at https://console.groq.com/dashboard/usage")
    print("Step 2: This script will run 1 full + 5 cancelled requests")
    print("Step 3: Check dashboard again after ~15 min to compare\n")

    # --- Baseline: one full request ---
    baseline_pt, baseline_ct = test_full_streaming()

    time.sleep(1)

    # --- 5 cancelled requests ---
    print("=" * 60)
    print("CANCELLED REQUESTS: 5x streaming, cancelled after ~10 chunks each")
    print("=" * 60)
    total_chunks = 0
    for i in range(5):
        elapsed, chunks, text, headers = test_cancelled_httpx(cancel_after_chunks=10)
        total_chunks += chunks
        print(f"  Run {i+1}: {elapsed:.2f}s, {chunks} content chunks, {len(text)} chars")

        # Check for any usage-related headers
        usage_headers = {k: v for k, v in headers.items()
                        if any(word in k.lower() for word in ["token", "usage", "billing", "rate", "remaining", "request"])}
        if usage_headers:
            print(f"         Relevant headers: {usage_headers}")
        time.sleep(1)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline (1 full request):     ~{baseline_pt} prompt + ~{baseline_ct} completion tokens")
    print(f"  Cancelled (5 requests x ~10 chunks): {total_chunks} total content chunks received")
    print()
    print("  If cancellation SAVES tokens:")
    print(f"    Expected billing: ~{baseline_pt * 5} prompt + ~{total_chunks} completion = ~{baseline_pt * 5 + total_chunks} total")
    print(f"  If cancellation does NOT save tokens:")
    print(f"    Expected billing: ~{baseline_pt * 5} prompt + ~{baseline_ct * 5} completion = ~{baseline_pt * 5 + baseline_ct * 5} total")
    print()
    print("  Check https://console.groq.com/dashboard/usage in ~15 min to see actual billing.")
    print("  Compare total token usage increase against these predictions.")
