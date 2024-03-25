from datasets import load_dataset, load_from_disk
from jiwer import wer, cer
from time import perf_counter
import asyncio
import argparse
import tritonclient.grpc.aio as async_grpcclient
import sys
import numpy as np
from tqdm.asyncio import tqdm_asyncio
from asr_utils import *
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="test_audio.wav",
        help="input audio file path",
    )

    parser.add_argument(
        "-u",
        "--url",
        default="server:8001",
        help="triton url",
    )

    parser.add_argument(
        "--data-dir",
        default="/data",
        help="data directory",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="wav2vec2",
        help="triton wav2vec2 model name",
    )

    parser.add_argument(
        "--vocab-path",
        type=str,
        default="vocab.json",
        help="huggingface vocab file path",
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=160000,
        help="audio max len",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="convert onnx model into fp16",
    )

    return parser.parse_args()


async def main():
    args = get_args()
    vocab_path = "/data/vocab.json"
    data_dir = Path("/data/zeroth_testset")
    if not data_dir.exists():
        ds = load_dataset("kresnik/zeroth_korean", "clean")["test"]
        ds.save_to_disk(data_dir)
    test_ds = load_from_disk(data_dir)
    vocab = get_vocab(vocab_path)

    try:
        triton_client = async_grpcclient.InferenceServerClient(url=args.url)

    except Exception as e:
        print(f"context creation failed: {e}")
        sys.exit()

    NUM_LOOPS = 3

    futures = [
        async_asr(audio_data["array"], triton_client, vocab, fp16=args.fp16)
        for _ in range(NUM_LOOPS)
        for audio_data in test_ds["audio"]
    ]

    ts1 = perf_counter()
    results = await tqdm_asyncio.gather(*futures)
    proc_time = perf_counter() - ts1
    proc_time = proc_time / NUM_LOOPS

    # cal metrics
    NUM_SAMPLES = test_ds.num_rows
    WER = wer(test_ds["text"], results[:NUM_SAMPLES])
    CER = cer(test_ds["text"], results[:NUM_SAMPLES])
    total_audio_len = get_total_audio_len(test_ds)

    RTF = proc_time / total_audio_len
    report_text = f"""\n\n------- BENCHMARK RESULT -------\nTOTAL PROC TIME: {proc_time:.2f} sec\nTOTAL AUDIO LEN: {total_audio_len:.2f} sec\nLOOP NUM: {NUM_LOOPS} \nWER: {WER * 100 :.2f} %\nCER: {CER * 100 :.2f} %\nRTF: {RTF:.4f}\nRTX: {1/RTF:.4f}\n"""
    sys.stdout.write(report_text)


if __name__ == "__main__":
    asyncio.run(main())
