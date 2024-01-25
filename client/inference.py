import tritonclient.grpc.aio as async_grpcclient

import sys
import soundfile as sf
import asyncio
import numpy as np
from scipy.io import wavfile
import scipy.signal as sps


URL = "localhost:8001"
VERBOSE = True
MODEL_NAME = "wav2vec2"
INPUT_NAME = "input"
OUTPUT_NAME = "output"
NEW_RATE = 16000
AUDIO_MAXLEN = 160000
WAV_PATH = "./export/test_audio.wav"


def layer_norm(data: np.array) -> np.array:
    avg = data.mean()
    var = data.var()
    return (data - avg) / (np.sqrt(var))


def _normalize(x):  #
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


def audio_prep(WAV_PATH):
    sampling_rate, data = wavfile.read(WAV_PATH)
    samples = round(len(data) * float(NEW_RATE) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = _normalize(speech)[None]
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)
    return speech


with open(f"./export/vocab.json", "r", encoding="utf-8") as f:
    d = eval(f.read())

res = dict((v, k) for k, v in d.items())


async def main():
    inputs = []
    outputs = []

    # y = np.array(data, dtype=np.float32)
    data = audio_prep(WAV_PATH)

    print(len(data[-1]))
    # y = layer_norm(y)

    try:
        triton_client = async_grpcclient.InferenceServerClient(url=URL)

    except Exception as e:
        print(f"context creation failed: {e}")
        sys.exit()

    print(triton_client)

    print("done")

    # inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data)], "FP32"))
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
    _t1 = "".join([res[i] for i in list(prediction[0])])
    final = "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])
    print(final)


if __name__ == "__main__":
    asyncio.run(main())
