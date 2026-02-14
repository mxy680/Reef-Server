"""
Test: Full voice tutoring pipeline latency

Pipeline: transcription → reasoning (GPT OSS 120B) → TTS (Kokoro)
Reasoning on Groq, TTS on DeepInfra. Measures each step and the pipelined version.
"""

import os, sys, time, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    sys.exit("GROQ_API_KEY not set")

deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
if not deepinfra_key:
    sys.exit("DEEPINFRA_API_KEY not set")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
tts_client = OpenAI(api_key=deepinfra_key, base_url="https://api.deepinfra.com/v1/openai")

# Simulate a transcription result (skip actual vision call for this test)
SAMPLE_TRANSCRIPTION = r"Find the derivative of f(x) = x^3 \sin(x)"

REASONING_MODEL = "openai/gpt-oss-120b"
REASONING_PROMPT = (
    "You are a math tutor. A student has written the following problem. "
    "Solve it step by step, speaking naturally as if explaining to the student. "
    "Keep your response concise (2-3 sentences for the key insight, then the answer).\n\n"
    f"Problem: {SAMPLE_TRANSCRIPTION}"
)

TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_VOICE = "af_bella"


def test_step1_transcription():
    """Simulate transcription step with a real Groq vision call."""
    print("STEP 1: Transcription (Groq Llama 4 Scout)")
    print("-" * 50)
    # Using a text prompt to simulate — in production this is a vision call
    t0 = time.time()
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": "Transcribe this math expression in LaTeX: 'find the derivative of f(x) = x cubed times sine of x'"}],
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    print(f"  Latency:  {elapsed:.2f}s")
    print(f"  Tokens:   {pt} in / {ct} out")
    print(f"  Result:   {text[:120]}")
    print()
    return elapsed, text


def test_step2_reasoning_full():
    """Full non-streaming reasoning — baseline."""
    print("STEP 2a: Reasoning - full (GPT OSS 120B)")
    print("-" * 50)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": REASONING_PROMPT}],
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    pt = resp.usage.prompt_tokens if resp.usage else 0
    ct = resp.usage.completion_tokens if resp.usage else 0
    print(f"  Latency:  {elapsed:.2f}s")
    print(f"  Tokens:   {pt} in / {ct} out")
    print(f"  Speed:    {ct / elapsed:.0f} tok/s")
    print(f"  Result:   {text[:200]}")
    print()
    return elapsed, text, ct


def test_step2_reasoning_streaming():
    """Streaming reasoning — measure time to first token and first sentence."""
    print("STEP 2b: Reasoning - streaming (GPT OSS 120B)")
    print("-" * 50)
    t0 = time.time()
    stream = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": REASONING_PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )

    first_token_time = None
    first_sentence_time = None
    first_sentence_text = ""
    full_text = ""
    token_count = 0
    usage = None

    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            token_count += 1
            full_text += content

            if first_token_time is None:
                first_token_time = time.time() - t0

            # Detect first sentence (ends with . or !)
            if first_sentence_time is None:
                first_sentence_text += content
                if any(first_sentence_text.rstrip().endswith(c) for c in ".!?"):
                    first_sentence_time = time.time() - t0

    elapsed = time.time() - t0
    ct = usage.completion_tokens if usage else token_count
    print(f"  First token:    {first_token_time:.3f}s")
    print(f"  First sentence: {first_sentence_time:.3f}s" if first_sentence_time else "  First sentence: N/A")
    print(f"  Total:          {elapsed:.2f}s")
    print(f"  Tokens:         {ct} out")
    print(f"  Speed:          {ct / elapsed:.0f} tok/s")
    print(f"  First sentence: \"{first_sentence_text.strip()[:150]}\"")
    print()
    return first_token_time, first_sentence_time, elapsed, first_sentence_text.strip(), full_text


def test_step3_tts(text):
    """TTS with streaming — measure time to first audio chunk."""
    text = text[:500]
    print(f"STEP 3: TTS - streaming Kokoro")
    print(f"  Input text ({len(text)} chars): \"{text[:100]}...\"")
    print("-" * 50)
    t0 = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0

    with tts_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    ) as response:
        for chunk in response.iter_bytes(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
            chunk_count += 1
            total_bytes += len(chunk)

    elapsed = time.time() - t0
    print(f"  First audio:  {first_chunk_time:.3f}s")
    print(f"  Total:        {elapsed:.2f}s")
    print(f"  Audio size:   {total_bytes:,} bytes")
    print()
    return first_chunk_time, elapsed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FULL VOICE TUTORING PIPELINE LATENCY TEST")
    print("Writing → Transcribe → Reason → Speak")
    print("=" * 60 + "\n")

    # Step 1: Transcription
    t1_latency, transcript = test_step1_transcription()

    time.sleep(0.5)

    # Step 2a: Full reasoning (baseline)
    t2_full_latency, reasoning_text, reasoning_tokens = test_step2_reasoning_full()

    time.sleep(0.5)

    # Step 2b: Streaming reasoning
    t2_first_tok, t2_first_sent, t2_total, first_sentence, full_reasoning = test_step2_reasoning_streaming()

    time.sleep(0.5)

    # Step 3a: TTS on full response
    t3_first_full, t3_total_full = test_step3_tts(reasoning_text)

    time.sleep(0.5)

    # Step 3b: TTS on just first sentence (pipelined)
    if first_sentence:
        t3_first_sent, t3_total_sent = test_step3_tts(first_sentence)
    else:
        t3_first_sent, t3_total_sent = 0, 0

    # Summary
    debounce = 1.0
    cluster = 0.03  # ~30ms average

    print("\n" + "=" * 60)
    print("LATENCY SUMMARY")
    print("=" * 60)

    sequential = debounce + cluster + t1_latency + t2_full_latency + t3_total_full
    print(f"\n  SEQUENTIAL (worst case):")
    print(f"    Debounce:        {debounce:.2f}s")
    print(f"    Clustering:      {cluster:.3f}s")
    print(f"    Transcription:   {t1_latency:.2f}s")
    print(f"    Reasoning:       {t2_full_latency:.2f}s")
    print(f"    TTS:             {t3_total_full:.2f}s")
    print(f"    ─────────────────────────")
    print(f"    TOTAL:           {sequential:.2f}s")

    pipelined = debounce + cluster + t1_latency + t2_first_sent + t3_first_sent
    print(f"\n  PIPELINED (first audio out):")
    print(f"    Debounce:        {debounce:.2f}s")
    print(f"    Clustering:      {cluster:.3f}s")
    print(f"    Transcription:   {t1_latency:.2f}s")
    print(f"    Reasoning→1st sentence: {t2_first_sent:.2f}s")
    print(f"    TTS first chunk: {t3_first_sent:.3f}s")
    print(f"    ─────────────────────────")
    print(f"    TOTAL:           {pipelined:.2f}s")

    print(f"\n  Speedup: {sequential / pipelined:.1f}x faster to first audio with pipelining")
    print()
