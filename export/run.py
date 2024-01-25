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
        "-o",
        "--output-dir",
        default="/models/wav2vec2/1/",
        help="onnx model export directory",
    )

    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="kresnik/wav2vec2-large-xlsr-korean",
        help="huggingface model name",
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
        "--max-len",
        type=int,
        default=160000,
        help="audio max len",
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


def asr(wav_path, ort_session, vocab):
    sampling_rate, data = wavfile.read(wav_path)
    samples = round(len(data) * float(16000) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = layer_norm(speech)[None]
    ort_inputs = {"input": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)
    _t1 = "".join([vocab[i] for i in list(prediction[0][0])])

    return "".join([remove_adjacent(j) for j in _t1.split("[PAD]")])


def main(args):
    org = Wav2Vec2ForCTC.from_pretrained(args.repo)
    hf_model = import_huggingface_model(org)
    hf_model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # model export
    dummy_input = torch.randn(1, args.max_len, requires_grad=True)
    torch.onnx.export(
        hf_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        f"{output_dir}/model.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=14,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 1: "input_sequence"},
            "output": {0: "batch_size", 1: "output_sequence"},
        },
    )

    # inference test
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        d = eval(f.read())
        vocab = dict((v, k) for k, v in d.items())

    ort_session = onnxruntime.InferenceSession(f"{output_dir}/model.onnx")  # load onnx model

    decoding_text = asr(args.wav_path, ort_session, vocab)
    decoding_text = decoding_text.replace("|", " ")

    sys.stdout.write(
        f"\nSuccess to export {args.repo} model !\noutput dir: {output_dir}\ntest decoding result: {decoding_text} \n"
    )


if __name__ == "__main__":
    args = getArgs()
    main(args)
