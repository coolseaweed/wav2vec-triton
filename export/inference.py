import argparse
from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model
import torch.onnx
from pathlib import Path
import sys
from scipy.io import wavfile
import scipy.signal as sps
import numpy as np
import onnx
import onnxruntime


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/models/wav2vec2/1/model.onnx",
        help="onnx model export directory",
    )

    parser.add_argument(
        "--vocab-path",
        type=str,
        default="/data/vocab.json",
        help="huggingface vocab file path",
    )

    parser.add_argument(
        "--wav-path",
        type=str,
        default="/data/test_audio.wav",
        help="test audio path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="convert onnx model into fp16",
    )

    return parser.parse_args()


def remove_adjacent(item):
    """code from https://stackoverflow.com/a/3460423"""
    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return "".join(a)


def layer_norm(x: np.array) -> np.array:
    """layer normalization
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return np.squeeze((x - mean) / (np.sqrt(var + 1e-5)))


def asr(wav_path, ort_session, vocab, fp16=False):
    data_type = np.float32 if not fp16 else np.float16

    sampling_rate, data = wavfile.read(wav_path)
    samples = round(len(data) * float(16000) / sampling_rate)
    speech = sps.resample(data, samples)
    speech = layer_norm(speech)[None].astype(data_type)
    ort_inputs = {"input": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)
    _t1 = "".join([vocab[i] for i in list(prediction[0][0])])

    return "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])


def main(args):
    # inference test
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        d = eval(f.read())
        vocab = dict((v, k) for k, v in d.items())

    ort_session = onnxruntime.InferenceSession(args.model_path)  # load onnx model

    decoding_text = asr(args.wav_path, ort_session, vocab, args.fp16)
    decoding_text = decoding_text.replace("|", " ")

    sys.stdout.write(f"decoding result: {decoding_text} \n")


if __name__ == "__main__":
    args = getArgs()
    main(args)
