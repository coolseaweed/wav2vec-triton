import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as async_grpcclient
from scipy.io import wavfile
import scipy.signal as sps

VERBOSE = True
INPUT_NAME = "input"
OUTPUT_NAME = "output"
NEW_RATE = 16000
AUDIO_MAXLEN = 160000


def layer_norm(x):  #
    """You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def get_total_audio_len(ds):
    total_audio_len = 0
    for audio_data in ds["audio"]:
        total_audio_len += len(audio_data["array"]) / audio_data["sampling_rate"]

    return total_audio_len


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


def audio_prep(wav_path, fp16=False):
    data_type = np.float32 if not fp16 else np.float16
    sampling_rate, data = wavfile.read(wav_path)
    samples = round(len(data) * float(NEW_RATE) / sampling_rate)
    speech = sps.resample(data, samples)
    speech = layer_norm(speech)[None].astype(data_type)
    return speech


def get_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        d = eval(f.read())

    return dict((v, k) for k, v in d.items())


async def async_asr(wav_data, triton_client, vocab, model_name="wav2vec2", fp16=False):
    inputs = []
    outputs = []

    data_type = "FP32" if not fp16 else "FP16"
    _data_type = np.float32 if not fp16 else np.float16

    data = layer_norm(wav_data)[None].astype(_data_type)
    inputs.append(async_grpcclient.InferInput(INPUT_NAME, [1, len(data[-1])], data_type))
    inputs[-1].set_data_from_numpy(data.reshape((1, len(data[-1]))))
    outputs.append(async_grpcclient.InferRequestedOutput(OUTPUT_NAME))
    result = await triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )

    result = result.as_numpy(OUTPUT_NAME)
    prediction = np.argmax(result, axis=-1).tolist()
    _t1 = "".join([vocab[i] for i in list(prediction[0])])
    trans = "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])

    return post_process(trans)
