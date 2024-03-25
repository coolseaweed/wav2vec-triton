from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, load_from_disk
import soundfile as sf
import torch
from jiwer import wer, cer
from time import perf_counter
import sys
import asyncio
from tqdm.asyncio import tqdm_asyncio
import nvidia_smi
from pathlib import Path


nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def get_total_audio_len(ds):
    total_audio_len = 0
    for audio_data in ds["audio"]:
        total_audio_len += len(audio_data["array"]) / audio_data["sampling_rate"]

    return total_audio_len


async def asr(wav_data, processor, model):
    inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


async def main():
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")

    model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean").to("cuda")

    data_dir = Path("/data/zeroth_testset")
    if not data_dir.exists():
        ds = load_dataset("kresnik/zeroth_korean", "clean")["test"]
        ds.save_to_disk(data_dir)
    test_ds = load_from_disk(data_dir)

    NUM_LOOPS = 3

    futures = [asr(audio_data["array"], processor, model) for _ in range(NUM_LOOPS) for audio_data in test_ds["audio"]]

    ts1 = perf_counter()
    results = await tqdm_asyncio.gather(*futures)
    proc_time = perf_counter() - ts1
    proc_time = proc_time / NUM_LOOPS

    # cal metrics
    NUM_SAMPLES = test_ds.num_rows
    WER = wer(test_ds["text"], results[:NUM_SAMPLES])
    CER = cer(test_ds["text"], results[:NUM_SAMPLES])
    total_audio_len = get_total_audio_len(test_ds)
    _info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    gpu_mem = (_info.used - info.used) / (2**20)

    RTF = proc_time / total_audio_len
    report_text = f"""\n\n------- BENCHMARK RESULT -------\nTOTAL PROC TIME: {proc_time:.2f} sec\nTOTAL AUDIO LEN: {total_audio_len:.2f} sec\nLOOP NUM: {NUM_LOOPS} \nWER: {WER * 100 :.2f} %\nCER: {CER * 100 :.2f} %\nRTF: {RTF:.4f}\nRTX: {1/RTF:.4f}\nGPU MEMORY USAGE: {gpu_mem} MiB\n"""
    sys.stdout.write(report_text)


if __name__ == "__main__":
    asyncio.run(main())
