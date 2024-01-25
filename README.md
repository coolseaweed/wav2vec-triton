# wav2vec2-triton

The purpose of repository is for exporting `wav2vec2` of huggingface into ONNX and depolying via TritonInferenceServers. The reference model of `wav2vec2` is based on 
[kresnik/wav2vec2-large-xlsr-korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean) model which is trained korean syllables.

## ONNX model export

- env setup
    ```
    docker compose --profile export up -d --build
    ```

- model export
    ```
    docker exec wav2vec2-triton-export-1 python run.py
    ```

## TritonInferenceServer

- env setup
    ```
    docker compose --profile server --profile client up -d --build
    ```
- inference test
    ```
    docker exec wav2vec2-triton-client-1 python inference.py -i test_audio.wav
    ```


## Benchmarking
For benchmarking, I used korean opean source dataset, [zeroth dataset](https://www.openslr.org/40/), to verify performance of ONNX+Triton combination. What I used spec of hardware is following.
- GPU: GTX 1080
- RAM: 16GB
- Ubuntu: 22.04
- Nvidia-Docker (GPU available)
- Nvidia driver version: 535.154.05

### How to benchmark
```bash
docker exec -it wav2vec2-triton-export-1 python benchmark.py # huggingface benchmark
docker exec -it wav2vec2-triton-client-1 python benchmark.py # triton benchmark
```

### Result
- dataset: `zeroth` clean testset
- Total audio length: 4287 sec

||huggingface|ONNX+Triton|
|:---|:---:|:---:|
|WER (%)|4.74|4.73|
|CER (%)|1.78|1.78|
|proc time (sec)|49.96|41.53|
|RTF|0.0115|0.0097|
|GPU (MiB)|3916|3286|