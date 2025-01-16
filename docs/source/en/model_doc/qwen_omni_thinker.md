<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# QwenOmniThinker

## Overview

The Qwen-Omni-Thinker is the new model series of large audio-language models from the Qwen team, which is the Thinker model of Qwen-Omni. Qwen-Omni can perceive all modalities including text, images, audio, and video, and simultaneously generate text and natural speech responses in a streaming manner.

For more details refer to the [release blog post](https://qwenlm.github.io/blog/qwen-moe/).

The abstract from the paper is the following:

*In this report, we introduce Qwen-Omni, a 7B parameter end-to-end single model that can perceive all modalities including text, images, audio, and video, and simultaneously generate text and natural speech responses in a streaming manner. For understanding, to support multimodal long time sequences and decoding pre-filling, both the audio and visual encoders employ a block-wise processing approach, thereby separating the perception capabilities from the long sequence understanding between the encoder and the LLM decoder. For video inputs with audio, we propose a time-interleaving method for audio and video, along with a novel position encoding approach, named ATMRoPE~(Absolute Time encoded Multimodal RoPE). To simultaneously generate text and speech while avoiding interference between the two, we propose \textbf{Thinker-Talker} architecture, where Thinker is a large language model responsible for producing text, while Talker is a dual-track autoregressive model that directly inherits the hidden representations from Thinker and streams audio tokens as output. Both models are trained and inferred in an end-to-end manner. Unlike previous cascaded text-to-speech (TTS) works, Talker directly inherits the hidden inputs from Thinker, which include images, videos, speech, and text, functioning more like a contextual speech dialogue model and theoretically possessing a higher capability ceiling. During inference, we introduce a sliding-window DiT model that limits the receptive field to decode speech codes streammly, aiming to achieve the lowest possible first package delay. *


## Usage tips

`Qwen-Omni-Thinker-7B` can be found on the [Huggingface Hub](https://huggingface.co/Qwen)

In the following, we demonstrate how to use `Qwen-Omni-Thinker-7B` for the inference, supporting all modalities including text, images, audio, and video. Note that we have used the ChatML format for dialog, in this demo we show how to leverage `apply_chat_template` for this purpose.

```python

from PIL import Image
import requests
import librosa
import torch
from torchvision import io
from typing import Dict
from transformers import QwenOmniThinkerForConditionalGeneration, AutoTokenizer, AutoProcessor

# Load the model in half-precision on the available device(s)
model = QwenOmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen-Omni-7B", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen-Omni-7B")

# Image
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)

# Video
def fetch_video(ele: Dict, nframe_factor=2):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        video, _, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], nframe_factor)
        else:
            fps = ele.get("fps", 1.0)
            nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)
        idx = torch.linspace(0, video.size(0) - 1, nframes, dtype=torch.int64)
        return video[idx]

video_info = {"type": "video", "video": "/path/to/video.mp4", "fps": 2.0}
audio = librosa.load(BytesIO(urlopen("/path/to/video.mp4").read()), sr=processor.feature_extractor.sampling_rate)[0]
video = fetch_video(video_info)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": "What happened in the video?"},
        ],
    }
]

# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text_prompt], videos=[video], audios=[audio], padding=True, return_tensors="pt", use_audio_in_video=True) #Use audio in video
inputs = inputs.to('cuda')

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)

# Audio
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
audio = librosa.load(BytesIO(urlopen(url).read()), sr=processor.feature_extractor.sampling_rate)[0]

conversation = [
    {
        "role":"user",
        "content":[
            {
                "type": "audio_url"
            }
        ]
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text_prompt], audios=[audio], padding=True, return_tensors="pt")
inputs = inputs.to('cuda')

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(output_text)
```

## QwenOmniThinkerAudioEncoderConfig

[[autodoc]] QwenOmniThinkerAudioEncoderConfig

## QwenOmniThinkerVisionEncoderConfig

[[autodoc]] QwenOmniThinkerVisionEncoderConfig

## QwenOmniThinkerConfig

[[autodoc]] QwenOmniThinkerConfig

## QwenOmniThinkerProcessor

[[autodoc]] QwenOmniThinkerProcessor

## QwenOmniThinkerModel

[[autodoc]] QwenOmniThinkerModel

## QwenOmniThinkerForConditionalGeneration

[[autodoc]] QwenOmniThinkerForConditionalGeneration
