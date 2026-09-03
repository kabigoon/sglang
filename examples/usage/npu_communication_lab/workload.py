#!/usr/bin/env python3
"""Send exact-length token requests to an SGLang /generate endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import statistics
import time

import requests
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:20766")
    parser.add_argument(
        "--model-path",
        default="/home/weights/DeepSeek-R1-0528-w4a8-per-channel",
    )
    parser.add_argument("--prompt-tokens", type=int, default=6144)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout", type=int, default=9000)
    return parser.parse_args()


def build_input_ids(model_path: str, length: int) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    pattern = tokenizer.encode(
        "请逐步分析这段用于分布式通信实验的长上下文。",
        add_special_tokens=False,
    )
    if not pattern:
        raise RuntimeError("tokenizer produced an empty token pattern")
    ids = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
    if tokenizer.bos_token_id is not None and length > 0:
        ids[0] = tokenizer.bos_token_id
    return ids


def post_one(url: str, ids: list[int], output_tokens: int, timeout: int, index: int):
    begin = time.perf_counter()
    response = requests.post(
        f"{url}/generate",
        json={
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    return index, time.perf_counter() - begin, body.get("meta_info", {})


def main() -> None:
    args = parse_args()
    if min(
        args.prompt_tokens,
        args.output_tokens,
        args.requests,
        args.concurrency,
        args.timeout,
    ) <= 0:
        raise ValueError(
            "token counts, request counts, concurrency, and timeout must be positive"
        )
    ids = build_input_ids(args.model_path, args.prompt_tokens)
    health = requests.get(f"{args.url}/health", timeout=10)
    health.raise_for_status()
    if args.profile:
        requests.post(
            f"{args.url}/start_profile", json={}, timeout=30
        ).raise_for_status()

    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            futures = [
                pool.submit(
                    post_one,
                    args.url,
                    ids,
                    args.output_tokens,
                    args.timeout,
                    index,
                )
                for index in range(args.requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                index, latency, meta = future.result()
                results.append(latency)
                print(f"request={index} latency={latency:.3f}s meta_info={meta}")
    finally:
        if args.profile:
            requests.post(f"{args.url}/stop_profile", timeout=30).raise_for_status()

    print(
        f"count={len(results)} mean={statistics.mean(results):.3f}s "
        f"min={min(results):.3f}s max={max(results):.3f}s"
    )


if __name__ == "__main__":
    main()
