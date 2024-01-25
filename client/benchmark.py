from datasets import load_dataset, load_from_disk
import soundfile as sf
from jiwer import wer, cer
from time import perf_counter
import asyncio
import argparse
import tritonclient.grpc.aio as async_grpcclient
import sys
import numpy as np
from tqdm.asyncio import tqdm_asyncio


INPUT_NAME = "input"
OUTPUT_NAME = "output"
MODEL_NAME = "wav2vec2"


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

    return parser.parse_args()


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def get_total_audio_len(ds):
    total_audio_len = 0
    for audio_data in ds["audio"]:
        total_audio_len += len(audio_data["array"]) / audio_data["sampling_rate"]

    return total_audio_len


def layer_norm(x):  #
    """You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5)).astype("f4")


def get_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        d = eval(f.read())

    return dict((v, k) for k, v in d.items())


def remove_adjacent(item):
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return "".join(a)


def post_process(text):
    text = text.replace("|", " ")

    return text


async def asr(wav_data, triton_client, vocab):
    inputs = []
    outputs = []

    data = layer_norm(wav_data)[None]
    inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data[-1])], "FP32"))
    inputs[-1].set_data_from_numpy(data.reshape((1, len(data[-1]))))
    outputs.append(async_grpcclient.InferRequestedOutput(OUTPUT_NAME))
    result = await triton_client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    result = result.as_numpy(OUTPUT_NAME)
    prediction = np.argmax(result, axis=-1).tolist()
    _t1 = "".join([vocab[i] for i in list(prediction[0])])
    trans = "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])

    return post_process(trans)


async def main():
    args = get_args()
    vocab_path = "/data/vocab.json"
    test_ds = load_from_disk("/data/zeroth_testset")

    vocab = get_vocab(vocab_path)

    try:
        triton_client = async_grpcclient.InferenceServerClient(url=args.url)

    except Exception as e:
        print(f"context creation failed: {e}")
        sys.exit()

    NUM_LOOPS = 3

    futures = [
        asr(audio_data["array"], triton_client, vocab) for _ in range(NUM_LOOPS) for audio_data in test_ds["audio"]
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
