"""
Test: Can we stream audio output from Kokoro TTS on DeepInfra?

Tests both non-streaming and streaming approaches.
"""

import os, sys, time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DEEPINFRA_API_KEY")
if not api_key:
    sys.exit("DEEPINFRA_API_KEY not set — add it to .env")

MODEL = "hexgrad/Kokoro-82M"
TEXT = "Welcome to Kokoro text-to-speech. This is a test of streaming audio output from DeepInfra."
VOICE = "af_bella"


def test_non_streaming():
    """Standard non-streaming TTS request."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    print("=" * 60)
    print("TEST 1: Non-streaming TTS")
    print("=" * 60)
    t0 = time.time()
    response = client.audio.speech.create(
        model=MODEL,
        voice=VOICE,
        input=TEXT,
        response_format="wav",
    )
    elapsed = time.time() - t0

    out_file = "/tmp/kokoro_non_stream.wav"
    response.write_to_file(out_file)
    file_size = os.path.getsize(out_file)

    print(f"  Time:      {elapsed:.2f}s")
    print(f"  File size: {file_size:,} bytes")
    print(f"  Saved to:  {out_file}")
    print()
    return elapsed, file_size


def test_streaming_openai():
    """Streaming TTS using OpenAI client's with_streaming_response."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")

    print("=" * 60)
    print("TEST 2: Streaming TTS (OpenAI client with_streaming_response)")
    print("=" * 60)
    t0 = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0

    with client.audio.speech.with_streaming_response.create(
        model=MODEL,
        voice=VOICE,
        input=TEXT,
        response_format="wav",
    ) as response:
        out_file = "/tmp/kokoro_stream_openai.wav"
        with open(out_file, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=4096):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - t0
                chunk_count += 1
                total_bytes += len(chunk)
                f.write(chunk)

    elapsed = time.time() - t0
    print(f"  Time to first chunk: {first_chunk_time:.3f}s")
    print(f"  Total time:          {elapsed:.2f}s")
    print(f"  Chunks received:     {chunk_count}")
    print(f"  Total bytes:         {total_bytes:,}")
    print(f"  Saved to:            {out_file}")
    print()
    return first_chunk_time, elapsed, total_bytes


def test_streaming_httpx():
    """Streaming TTS using raw httpx for maximum control."""
    import httpx

    print("=" * 60)
    print("TEST 3: Streaming TTS (raw httpx)")
    print("=" * 60)
    url = "https://api.deepinfra.com/v1/openai/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL,
        "voice": VOICE,
        "input": TEXT,
        "response_format": "wav",
    }

    t0 = time.time()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0
    response_headers = {}

    with httpx.Client(timeout=30) as http:
        with http.stream("POST", url, headers=headers, json=body) as resp:
            response_headers = dict(resp.headers)
            out_file = "/tmp/kokoro_stream_httpx.wav"
            with open(out_file, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=4096):
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - t0
                    chunk_count += 1
                    total_bytes += len(chunk)
                    f.write(chunk)

    elapsed = time.time() - t0
    print(f"  Time to first chunk: {first_chunk_time:.3f}s" if first_chunk_time else "  No chunks received")
    print(f"  Total time:          {elapsed:.2f}s")
    print(f"  Chunks received:     {chunk_count}")
    print(f"  Total bytes:         {total_bytes:,}")
    print(f"  Saved to:            {out_file}")
    print(f"  Content-Type:        {response_headers.get('content-type', 'N/A')}")
    print(f"  Transfer-Encoding:   {response_headers.get('transfer-encoding', 'N/A')}")
    print()

    # Show relevant headers
    relevant = {k: v for k, v in response_headers.items()
                if any(w in k.lower() for w in ["content", "transfer", "encoding", "length"])}
    if relevant:
        print(f"  Relevant headers: {relevant}")
        print()

    return first_chunk_time, elapsed, total_bytes


if __name__ == "__main__":
    print("\nKokoro TTS Streaming Test")
    print("=" * 60)

    r1_time, r1_size = test_non_streaming()
    time.sleep(1)

    try:
        r2_first, r2_time, r2_bytes = test_streaming_openai()
    except Exception as e:
        print(f"  ERROR: {e}\n")
        r2_first, r2_time, r2_bytes = None, None, None

    time.sleep(1)

    try:
        r3_first, r3_time, r3_bytes = test_streaming_httpx()
    except Exception as e:
        print(f"  ERROR: {e}\n")
        r3_first, r3_time, r3_bytes = None, None, None

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Non-streaming:       {r1_time:.2f}s, {r1_size:,} bytes")
    if r2_first is not None:
        print(f"  Streaming (OpenAI):  first chunk {r2_first:.3f}s, total {r2_time:.2f}s, {r2_bytes:,} bytes")
    if r3_first is not None:
        print(f"  Streaming (httpx):   first chunk {r3_first:.3f}s, total {r3_time:.2f}s, {r3_bytes:,} bytes")
    print()
    if r2_first and r2_first < r1_time * 0.5:
        print("  ✓ Streaming delivers first audio chunk significantly faster!")
    else:
        print("  Streaming may not provide a meaningful latency improvement.")
