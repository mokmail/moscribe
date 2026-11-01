# Third-party components

MoScribe is MIT licensed. It runs other people's models and code, and their terms
apply to those parts.

| Component | Used for | Licence | Source |
| --- | --- | --- | --- |
| VibeASR.cpp | ASR inference server (`asr_stream_server`) | MIT | https://github.com/microsoft/VibeASR.cpp |
| VibeVoice-ASR-BitNet (VAE + LM GGUF) | speech recognition weights | MIT (per model card) | https://huggingface.co/microsoft/VibeVoice-ASR-BitNet |
| llama.cpp (BitNet fork) | text inference (`llama-server`) | MIT | https://github.com/microsoft/BitNet |
| BitNet b1.58 2B-4T GGUF | summaries, minutes, rewrite | MIT (per model card) | https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf |
| WeSpeaker ResNet34-LM | speaker embeddings | Apache-2.0 (upstream WeSpeaker) | https://huggingface.co/onnx-community/wespeaker-voxceleb-resnet34-LM |
| ONNX Runtime | running the speaker model | MIT | https://github.com/microsoft/onnxruntime |
| FastAPI / Uvicorn | local HTTP + WebSocket API | MIT / BSD-3 | https://fastapi.tiangolo.com |
| Inter typeface | interface type | SIL OFL 1.1 | https://rsms.me/inter |

Model weights are **not** included in this repository or in the installers; they are
downloaded on first run from Hugging Face by `moscribe --fetch-models`.

The kmail mark is © Mohammed / kmail.at and is not covered by the MIT grant.
