import tritonclient.grpc.aio as async_grpcclient
import argparse
import sys
import asyncio
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps
from pathlib import Path

VERBOSE = True
INPUT_NAME = "input"
OUTPUT_NAME = "output"
NEW_RATE = 16000
AUDIO_MAXLEN = 160000


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


def layer_norm(x):  #
    """You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def remove_adjacent(item):  # code from https://stackoverflow.com/a/3460423
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return "".join(a)


def audio_prep(wav_path):
    sampling_rate, data = wavfile.read(wav_path)
    samples = round(len(data) * float(NEW_RATE) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = layer_norm(speech)[None]
    return speech


def get_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        d = eval(f.read())

    return dict((v, k) for k, v in d.items())


async def main():
    inputs = []
    outputs = []

    args = get_args()
    data_dir = Path(args.data_dir)
    vocab_path = data_dir / Path(args.vocab_path)
    wav_path = data_dir / Path(args.input)

    if not wav_path.exists():
        raise FileExistsError(f"'{args.input}' file not exist in data directory")

    vocab = get_vocab(vocab_path)
    data = audio_prep(wav_path)

    try:
        triton_client = async_grpcclient.InferenceServerClient(url=args.url)

    except Exception as e:
        print(f"context creation failed: {e}")
        sys.exit()

    # inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data)], "FP32"))
    inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data[-1])], "FP32"))
    inputs[-1].set_data_from_numpy(data.reshape((1, len(data[-1]))))
    outputs.append(async_grpcclient.InferRequestedOutput(OUTPUT_NAME))

    result = await triton_client.infer(
        model_name=args.model_name,
        inputs=inputs,
        outputs=outputs,
    )

    result = result.as_numpy(OUTPUT_NAME)
    prediction = np.argmax(result, axis=-1).tolist()
    _t1 = "".join([vocab[i] for i in list(prediction[0])])
    final = "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])
    print(final)


if __name__ == "__main__":
    asyncio.run(main())
