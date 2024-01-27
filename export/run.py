import argparse
from transformers import Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model
import torch.onnx
from pathlib import Path
import onnx
from onnxconverter_common import float16
import shutil
import sys


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-dir",
        default="/models/wav2vec2",
        help="onnx model export directory",
    )

    parser.add_argument(
        "--version",
        type=str,
        default="1",
        help="export version",
    )

    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="kresnik/wav2vec2-large-xlsr-korean",
        help="huggingface model name",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="convert onnx model into fp16",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data",
        help="data directory path",
    )

    return parser.parse_args()


def hf2onnx(hf_model, output_dir, fp16=False):
    temp_dir = Path(".temp")
    temp_dir.mkdir(exist_ok=True)

    # model export
    dummy_input = torch.randn(1, 160000, requires_grad=True)
    torch.onnx.export(
        hf_model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        f"{temp_dir}/model.onnx",  # where to save the model
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

    if fp16:
        model_fp32 = onnx.load_model(f"{temp_dir}/model.onnx")
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        onnx.save(model_fp16, f"{output_dir}/model.onnx")
    else:
        shutil.move(f"{temp_dir}/model.onnx", f"{output_dir}/model.onnx")

    shutil.rmtree(temp_dir)


def main(args):
    org = Wav2Vec2ForCTC.from_pretrained(args.repo)
    hf_model = import_huggingface_model(org)
    hf_model.eval()

    model_out_dir = Path(f"{args.output_dir}/{args.version}")
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # export model
    hf2onnx(hf_model, model_out_dir, args.fp16)

    # copy config.pbtxt
    config_file = "config_fp16.pbtxt" if args.fp16 else "config_fp32.pbtxt"
    pbtxt_path = Path(f"{args.data_dir}/{config_file}")
    if not pbtxt_path.exists():
        raise FileNotFoundError(f"{pbtxt_path} not exist !")

    shutil.copyfile(pbtxt_path, f"{args.output_dir}/config.pbtxt")

    # final report
    precision = "FP32" if not args.fp16 else "FP16"
    sys.stdout.write(
        f"\nSuccess to export {args.repo} model !\noutput dir: {args.output_dir}\nPrecision: {precision}\n"
    )


if __name__ == "__main__":
    args = getArgs()
    main(args)
