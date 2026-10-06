"""Mixed closed-loop client for the S5 A/B: streaming-audio requests, plus optional text-only long-prompt load.

Both loads go to ONE mstar server (Qwen3-Omni, `configs/qwen3omni_colocated.yaml`) from ONE asyncio loop, using the
repo's own `benchmark.request.OurSystem` adapter, so request bytes and timestamps are exactly what `benchmark.runner`
produces. `benchmark.runner` itself runs one request type per process and reports only aggregates, which is why this
wrapper exists: it keeps every request's chunk arrival times, so the analysis can compute stalls and per-token ITL.

  audio load: N text_to_speech requests, closed loop at --audio-conc. These are the measured requests.
  text load : text_to_text requests with 1-2k-token prompts, closed loop at --text-conc, refilled until the audio load
              has finished, so the contention lasts the whole measured window (in-flight ones then drain).

Output: one JSON, with per-request rows (monotonic times converted to wall clock, so they join the server's telemetry
`ts`) and the window [t_start, t_end] of the audio load.

Run from anything, with PYTHONPATH=<mstar worktree> (the `benchmark` package is not installed):
    PYTHONPATH=$MSTAR_DIR $MSTAR_DIR/.venv/bin/python mixed_client.py --url http://127.0.0.1:PORT --out case/client.json \
        --audio-conc 4 --audio-requests 24 --text-conc 8 --text-prompts prompts_long_1k2k.txt --tag b4_r1
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp

import benchmark
from benchmark.base import ModelType, RequestType
from benchmark.request import OurSystem, RequestInput, RequestMetrics

SAMPLE_BYTES_PER_S = 24000 * 2  # 24 kHz int16 mono, what Code2Wav emits


def audio_row(m: RequestMetrics, wall_off: float) -> dict:
    arrivals = m.chunk_arrivals.get("audio", [])
    nbytes = m.output_bytes.get("audio", 0)
    dur = nbytes / SAMPLE_BYTES_PER_S
    # Playback model: the listener starts at the first chunk and plays at 1x. A chunk that arrives after the audio
    # buffered so far ran out is a stall of that length. RequestMetrics keeps (gap, duration of the previous chunk)
    # for every chunk after the first, plus the last chunk's duration, which is every chunk's duration.
    n = len(arrivals)
    durs = [prev for _gap, prev in m._audio_chunk_log]
    if m._last_audio_chunk_duration is not None:
        durs.append(m._last_audio_chunk_duration)
    stall, played_until = 0.0, None
    if len(durs) == n:
        for t, d in zip(arrivals, durs):
            if played_until is None:
                played_until = t
            elif t > played_until:
                stall += t - played_until
                played_until = t
            played_until += d
    return {
        "ok": m.error is None and m.e2e_latency is not None and n > 0,
        "error": m.error,
        "start": (m.start_time or 0.0) + wall_off,
        "ttfa": m.ttft.get("audio"),
        "e2e": m.e2e_latency,
        "audio_s": dur,
        "rtf": (m.e2e_latency / dur) if dur > 0 and m.e2e_latency else None,
        "n_chunks": n,
        "stall_s": stall,
        "gaps": [b - a for a, b in zip(arrivals, arrivals[1:])],
        "end": ((m.start_time or 0.0) + (m.e2e_latency or 0.0)) + wall_off,
    }


def text_row(m: RequestMetrics, wall_off: float) -> dict:
    arrivals = m.chunk_arrivals.get("text", [])
    return {
        "ok": m.error is None and m.e2e_latency is not None and len(arrivals) > 0,
        "error": m.error,
        "start": (m.start_time or 0.0) + wall_off,
        "ttft": m.ttft.get("text"),
        "e2e": m.e2e_latency,
        "n_tokens": m.output_text_tokens,
        "gaps": [b - a for a, b in zip(arrivals, arrivals[1:])],
        "end": ((m.start_time or 0.0) + (m.e2e_latency or 0.0)) + wall_off,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="x")
    ap.add_argument("--audio-conc", type=int, required=True)
    ap.add_argument("--audio-requests", type=int, required=True)
    ap.add_argument(
        "--audio-prompts",
        default=os.path.join(os.path.dirname(benchmark.__file__), "assets", "simple_text_queries.txt"),
    )
    ap.add_argument("--audio-max-tokens", type=int, default=64, help="Thinker cap for the spoken answer")
    ap.add_argument("--text-conc", type=int, default=0)
    ap.add_argument("--text-prompts", default=None, help="one long prompt per line (make_long_prompts.py)")
    ap.add_argument("--text-max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    model = ModelType.QWEN3OMNI.inst()
    system = OurSystem()

    with open(args.audio_prompts) as f:
        a_prompts = [ln.strip() for ln in f if ln.strip()]
    audio_reqs = []
    for i in range(args.audio_requests):
        audio_reqs.append(RequestInput(
            req_type=RequestType.T2S, prompt=a_prompts[i % len(a_prompts)],
            model_kwargs={"max_tokens": args.audio_max_tokens, "max_output_tokens": args.audio_max_tokens},
            request_id=f"{args.tag}-a{i}",
        ))
    t_prompts = []
    if args.text_conc > 0:
        with open(args.text_prompts) as f:
            t_prompts = [ln.rstrip("\n") for ln in f if ln.strip()]

    wall_off = time.time() - time.monotonic()
    audio_done = asyncio.Event()
    audio_out: list[RequestMetrics] = []
    text_out: list[RequestMetrics] = []
    window = {}

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        connector=aiohttp.TCPConnector(limit=args.audio_conc + args.text_conc + 10),
        read_bufsize=5 * 2**20,
    ) as session:

        async def audio_worker(queue: asyncio.Queue):
            while True:
                try:
                    i = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                audio_out.append(await system.send_request(
                    session=session, req_input=audio_reqs[i], base_url=args.url, request_id=i, model=model,
                ))

        async def text_worker(w: int):
            k = w
            while not audio_done.is_set():
                prompt = t_prompts[k % len(t_prompts)]
                k += args.text_conc
                req = RequestInput(
                    req_type=RequestType.T2T, prompt=prompt,
                    model_kwargs={
                        "max_tokens": args.text_max_tokens, "max_output_tokens": args.text_max_tokens,
                        "ignore_eos": True,
                    },
                    request_id=f"{args.tag}-t{w}-{k}",
                )
                text_out.append(await system.send_request(
                    session=session, req_input=req, base_url=args.url, request_id=10_000 + k, model=model,
                ))

        queue: asyncio.Queue = asyncio.Queue()
        for i in range(args.audio_requests):
            queue.put_nowait(i)
        window["t_start"] = time.time()
        text_tasks = [asyncio.create_task(text_worker(w)) for w in range(args.text_conc)]
        await asyncio.gather(*(audio_worker(queue) for _ in range(args.audio_conc)))
        window["t_end"] = time.time()
        audio_done.set()
        await asyncio.gather(*text_tasks)
        window["t_drained"] = time.time()

    out = {
        "tag": args.tag, "args": vars(args), "window": window, "wall_off": wall_off,
        "audio": [audio_row(m, wall_off) for m in audio_out],
        "text": [text_row(m, wall_off) for m in text_out],
    }
    with open(args.out + ".tmp", "w") as f:  # written whole or not at all: its existence marks the case done
        json.dump(out, f)
    os.replace(args.out + ".tmp", args.out)
    ok_a = [r for r in out["audio"] if r["ok"]]
    ok_t = [r for r in out["text"] if r["ok"]]
    dur = window["t_end"] - window["t_start"]
    print(
        f"[{args.tag}] window {dur:.1f}s  audio ok {len(ok_a)}/{len(out['audio'])}"
        f"  ttfa mean {sum(r['ttfa'] for r in ok_a) / max(len(ok_a), 1):.2f}s"
        f"  text ok {len(ok_t)}/{len(out['text'])}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
