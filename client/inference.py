import tritonclient.grpc.aio as async_grpcclient
import argparse
import sys
import asyncio
import numpy as np

from pathlib import Path
from asr_utils import *


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
    inputs = []
    outputs = []

    args = get_args()
    data_dir = Path(args.data_dir)
    vocab_path = data_dir / Path(args.vocab_path)
    wav_path = data_dir / Path(args.input)

    precision = "FP32" if not args.fp16 else "FP16"

    print(f"precision is : {precision}")

    if not wav_path.exists():
        raise FileExistsError(f"'{args.input}' file not exist in data directory")

    vocab = get_vocab(vocab_path)
    data = audio_prep(wav_path, args.fp16)

    try:
        triton_client = async_grpcclient.InferenceServerClient(url=args.url)

    except Exception as e:
        print(f"context creation failed: {e}")
        sys.exit()

    inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data[-1])], precision))
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
