"""
Test: Stream reasoning → sentence-buffer → parallel TTS requests

As GPT OSS 120B streams tokens, we buffer until a sentence boundary,
then immediately fire a TTS request for that sentence while continuing
to buffer the next one. Measures how this overlapping approach performs.
"""

import os, sys, time, re, asyncio, io
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    sys.exit("GROQ_API_KEY not set")

deepinfra_key = os.getenv("DEEPINFRA_API_KEY")
if not deepinfra_key:
    sys.exit("DEEPINFRA_API_KEY not set")

client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
async_client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
tts_client = OpenAI(api_key=deepinfra_key, base_url="https://api.deepinfra.com/v1/openai")

REASONING_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_VOICE = "af_bella"

PROMPT = (
    "You are a math tutor. A student has written the following problem. "
    "Solve it step by step, speaking naturally as if explaining to the student. "
    "Keep your response concise (2-3 sentences for the key insight, then the answer).\n\n"
    "Problem: Find the derivative of f(x) = x^3 sin(x)"
)

# Sentence boundary pattern — split on ., !, ? followed by space or end
SENTENCE_END = re.compile(r'[.!?](?:\s|$)')


def tts_for_sentence(sentence: str, index: int, t0: float) -> dict:
    """Fire a TTS request for one sentence, return timing info."""
    sentence = sentence.strip()
    if not sentence or len(sentence) < 5:
        return {"index": index, "skipped": True, "text": sentence}

    sentence = sentence[:500]

    start = time.time() - t0
    first_chunk_time = None
    total_bytes = 0

    with tts_client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=sentence,
        response_format="wav",
    ) as response:
        out_file = f"/tmp/kokoro_sentence_{index}.wav"
        with open(out_file, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=4096):
                if first_chunk_time is None:
                    first_chunk_time = time.time() - t0
                total_bytes += len(chunk)
                f.write(chunk)

    done = time.time() - t0
    return {
        "index": index,
        "text": sentence[:80],
        "tts_start": start,
        "first_audio": first_chunk_time,
        "tts_done": done,
        "bytes": total_bytes,
        "file": out_file,
    }


def run_pipeline():
    """Stream reasoning, sentence-buffer, fire TTS per sentence."""
    print("=" * 60)
    print("PIPELINED: Reasoning stream → sentence buffer → TTS")
    print("=" * 60)

    t0 = time.time()

    # Start streaming reasoning
    stream = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        stream=True,
        stream_options={"include_usage": True},
    )

    sentences = []
    buffer = ""
    first_token_time = None
    sentence_times = []  # (sentence_text, time_sentence_complete)

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.time() - t0
            buffer += content

            # Check for sentence boundary
            match = SENTENCE_END.search(buffer)
            if match:
                end_pos = match.end()
                sentence = buffer[:end_pos].strip()
                buffer = buffer[end_pos:]
                if sentence:
                    t_sent = time.time() - t0
                    sentences.append(sentence)
                    sentence_times.append((sentence, t_sent))

    # Don't forget remaining buffer
    if buffer.strip():
        sentences.append(buffer.strip())
        sentence_times.append((buffer.strip(), time.time() - t0))

    reasoning_done = time.time() - t0

    print(f"\n  Reasoning first token: {first_token_time:.3f}s")
    print(f"  Reasoning complete:    {reasoning_done:.2f}s")
    print(f"  Sentences found:       {len(sentences)}")
    for i, (sent, t) in enumerate(sentence_times):
        print(f"    [{i}] @{t:.3f}s: \"{sent[:80]}\"")
    print()

    # Now simulate the pipelined approach:
    # In production, you'd fire TTS as soon as each sentence is ready.
    # Here we measure sequential vs overlapped TTS timing.

    # --- Sequential TTS (one after another) ---
    print("-" * 60)
    print("TTS SEQUENTIAL (one sentence at a time):")
    print("-" * 60)
    seq_results = []
    for i, sent in enumerate(sentences):
        result = tts_for_sentence(sent, i, t0)
        seq_results.append(result)
        if not result.get("skipped"):
            print(f"  [{i}] first_audio={result['first_audio']:.3f}s  done={result['tts_done']:.3f}s  ({result['bytes']:,}B)")

    print()

    # --- Pipelined TTS (fire as sentences arrive) ---
    # Simulate: for each sentence, TTS starts at the time the sentence was complete
    print("-" * 60)
    print("PIPELINED TIMING (TTS fires when each sentence completes):")
    print("-" * 60)

    # Re-run TTS for each sentence and record durations
    tts_durations = []
    for i, sent in enumerate(sentences):
        t_start = time.time()
        first_audio_offset = None
        total_bytes = 0
        sent_text = sent.strip()[:500]
        if len(sent_text) < 5:
            continue
        with tts_client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=sent_text,
            response_format="wav",
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                if first_audio_offset is None:
                    first_audio_offset = time.time() - t_start
                total_bytes += len(chunk)
        total_dur = time.time() - t_start
        tts_durations.append({
            "index": i,
            "first_audio": first_audio_offset,
            "total": total_dur,
            "bytes": total_bytes,
        })

    # Calculate pipelined timeline
    # Each TTS starts when its sentence finishes from reasoning stream
    print()
    pipeline_first_audio = None
    prev_audio_end = 0
    for i, (dur, (sent, sent_time)) in enumerate(zip(tts_durations, sentence_times)):
        tts_start = sent_time  # TTS fires when sentence is ready
        first_audio_at = tts_start + dur["first_audio"]
        audio_done_at = tts_start + dur["total"]

        # Audio can only start playing after previous audio finishes
        actual_play_start = max(first_audio_at, prev_audio_end)

        if pipeline_first_audio is None:
            pipeline_first_audio = first_audio_at

        print(f"  Sentence {i}: ready@{sent_time:.3f}s → first_audio@{first_audio_at:.3f}s → done@{audio_done_at:.3f}s")

        # Estimate when this sentence's audio finishes playing
        # (audio duration ≈ total_bytes / 48000 / 2 for 16-bit 24kHz WAV)
        audio_duration = dur["bytes"] / (24000 * 2)  # rough estimate
        prev_audio_end = actual_play_start + audio_duration

    total_pipeline_time = time.time() - t0

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    debounce = 1.0
    cluster = 0.03
    transcription = 1.39  # from previous test

    print(f"\n  Fixed overhead:")
    print(f"    Debounce:          {debounce:.2f}s")
    print(f"    Clustering:        {cluster:.3f}s")
    print(f"    Transcription:     {transcription:.2f}s  (from previous test)")

    print(f"\n  Reasoning:")
    print(f"    First token:       {first_token_time:.3f}s")
    print(f"    First sentence:    {sentence_times[0][1]:.3f}s")
    print(f"    Full response:     {reasoning_done:.2f}s")

    print(f"\n  TTS (per sentence):")
    for d in tts_durations:
        print(f"    [{d['index']}] first_audio: {d['first_audio']:.3f}s, total: {d['total']:.2f}s")

    if pipeline_first_audio:
        total_to_first_audio = debounce + cluster + transcription + pipeline_first_audio
        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  TIME TO FIRST AUDIO (pipelined): {total_to_first_audio:.2f}s")
        print(f"    = {debounce}s debounce + {cluster}s cluster + {transcription}s transcribe")
        print(f"      + {sentence_times[0][1]:.3f}s first sentence + {tts_durations[0]['first_audio']:.3f}s TTS")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    run_pipeline()
