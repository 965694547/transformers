# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch QwenOmniThinker model."""

import tempfile
import unittest
from io import BytesIO
from urllib.request import urlopen

import librosa
import requests

from transformers import (
    AutoProcessor,
    QwenOmniThinkerConfig,
    QwenOmniThinkerForConditionalGeneration,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_deepspeed,
    require_torch,
    require_torch_gpu,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _deepspeed_zero3,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


class QwenOmniThinkerModelTester:
    def __init__(
        self,
        parent,
        audio_token_index=0,
        image_token_index=1,
        video_token_index=2,
        vision_config={
            "depth": 2,
            "embed_dim": 32,
            "hidden_act": "quick_gelu",
            "hidden_size": 32,
            "mlp_ratio": 4,
            "num_heads": 4,
            "patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "_attn_implementation": "flash_attention_2",
            "initializer_range": 0.02,
        },
        text_config={
            "model_type": "qwen2",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 100,
        },
        audio_config={
            "model_type": "qwen_omni_thinker_audio_encoder",
            "d_model": 16,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 16,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "max_source_positions": 30,
            "initializer_range": 0.02,
            "n_window": 10,
            "output_dim": 32,
            "_attn_implementation": "flash_attention_2",
        },
        batch_size=3,
        feat_seq_length=60,
        num_channels=3,
        image_size=14,
        seq_length=39,
        hidden_act="silu",
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        num_key_value_heads=2,
        is_training=True,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        attention_dropout=0.0,
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        position_id_per_seconds=25,
        seconds_per_chunk=2,
        audio_start_token_id=151647,
        audio_end_token_id=151648,
    ):
        self.parent = parent
        self.audio_token_index = audio_token_index
        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.vision_config = vision_config
        self.text_config = text_config
        self.audio_config = audio_config
        self.batch_size = batch_size
        self.feat_seq_length = feat_seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.num_image_tokens = 32
        self.seq_length = seq_length
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads
        self.is_training = is_training
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.position_id_per_seconds = position_id_per_seconds
        self.seconds_per_chunk = seconds_per_chunk
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.initializer_range = text_config["initializer_range"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return QwenOmniThinkerConfig(
            audio_config=self.audio_config,
            vision_config=self.vision_config,
            audio_token_index=self.audio_token_index,
            image_token_index=self.image_token_index,
            video_token_index=self.video_token_index,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            max_window_layers=self.max_window_layers,
            attention_dropout=self.attention_dropout,
            rope_scaling=self.rope_scaling,
            position_id_per_seconds=self.position_id_per_seconds,
            seconds_per_chunk=self.seconds_per_chunk,
            audio_start_token_id=self.audio_start_token_id,
            audio_end_token_id=self.audio_end_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        pixel_grid_thw = torch.LongTensor(
            [[1, self.image_size / patch_size, self.image_size / patch_size]] * self.batch_size
        ).to(pixel_values.device)
        input_features_values = floats_tensor(
            [
                self.audio_config["num_mel_bins"],
                self.feat_seq_length * self.batch_size,
            ]
        )
        feature_attention_mask = torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)
        return config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_grid_thw, input_features_values, feature_attention_mask = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size - 3) + 3
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        attention_mask[:, :1] = 0
        audio_feat_length = ((self.feat_seq_length - 1) // 2 + 1 - 2) // 2 + 1
        input_ids[:, 1 : (1 + audio_feat_length)] = config.audio_token_index
        input_ids[:, -2] = config.image_token_index
        inputs_dict = {
            "input_features": input_features_values,
            "feature_attention_mask": feature_attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_grid_thw": pixel_grid_thw,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict

    def create_and_check_qwenomnithinker_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = QwenOmniThinkerForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class QwenOmniThinkerForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `QwenOmniThinkerForConditionalGeneration`.
    """

    all_model_classes = (QwenOmniThinkerForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = QwenOmniThinkerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=QwenOmniThinkerConfig, has_text_modality=False)

    @unittest.skip(reason="Compile not yet supported because in QwenOmniThinker models")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in QwenOmniThinker models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because Qwen2 is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                audio_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.visual._supports_sdpa else "eager"
                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == audio_attn)
                self.assertTrue(model.visual.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")
                self.assertTrue(model_eager.visual.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    @require_deepspeed
    @require_torch_gpu
    def test_resize_tokens_embeddings_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
            "train_batch_size": 3,
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_tokens_embeddings()

    @require_deepspeed
    @require_torch_gpu
    def test_resize_embeddings_untied_with_deepspeed(self):
        ds_config = {
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
            "train_batch_size": 3,
        }
        with _deepspeed_zero3(ds_config):
            self.test_resize_embeddings_untied()


@require_torch
class QwenOmniThinkerForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen-Omni-Thinker-7B")
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
        image_url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg"
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": "What's that sound?"},
                    {"type": "image", "image_url": image_url},
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            }
        ]

        self.raw_audio, _ = librosa.load(
            BytesIO(urlopen(audio_url).read()), sr=self.processor.feature_extractor.sampling_rate
        )
        self.raw_image = Image.open(requests.get(image_url, stream=True).raw)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test_single(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = QwenOmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen-Omni-Thinker-7B")

        text = self.processor.apply_chat_template(self.messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text], audios=[self.raw_audio], images=[self.raw_image], return_tensors="pt", padding=True
        )

        output = model.generate(**inputs, max_new_tokens=64).sequences

        EXPECTED_INPUT_IDS = torch.tensor(
            [
                [
                    151644,
                    8948,
                    198,
                    2610,
                    525,
                    264,
                    10950,
                    17847,
                    13,
                    151645,
                    198,
                    151644,
                    872,
                    198,
                    151647,
                    151646,
                    151648,
                    3838,
                    594,
                    429,
                    5112,
                    30,
                    151652,
                    151655,
                    151653,
                    3838,
                    3093,
                    315,
                    5562,
                    374,
                    419,
                    30,
                    151645,
                    198,
                    151644,
                    77091,
                    198,
                ]
            ]
        )
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        EXPECTED_DECODED_TEXT = "system\nYou are a helpful assistant.\nuser\nWhat's that sound?What kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labrador Retrievers are known for their friendly and affectionate nature, and they are often used as family pets."

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=False),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_batch(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = QwenOmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen-Omni-Thinker-7B")

        text = self.processor.apply_chat_template(self.messages, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=[text, text],
            audios=[self.raw_audio, self.raw_audio],
            images=[self.raw_image, self.raw_image],
            return_tensors="pt",
            padding=True,
        )

        output = model.generate(**inputs, max_new_tokens=64).sequences

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound?What kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labrador Retrievers are known for their friendly and affectionate nature, and they are often used as family pets.",
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound?What kind of dog is this?\nassistant\nThe dog in the picture appears to be a Labrador Retriever. Labrador Retrievers are known for their friendly and affectionate nature, and they are often used as family pets.",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    def test_small_model_integration_test_multiturn(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model = QwenOmniThinkerForConditionalGeneration.from_pretrained("Qwen/Qwen-Omni-Thinker-7B")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3",
                    },
                    {"type": "text", "text": "What's that sound?"},
                ],
            },
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg",
                    },
                    {"type": "text", "text": "What kind of dog is this?"},
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text], audios=[self.raw_audio], images=[self.raw_images], return_tensors="pt", padding=True
        )

        output = model.generate(**inputs, max_new_tokens=64, top_k=1).sequences

        EXPECTED_DECODED_TEXT = [
            "system\nYou are a helpful assistant.\nuser\nWhat's that sound?\nassistant\nIt is the sound of glass shattering.\nuser\nWhat kind of dog is this?\nassistant\nThis is a Labrador Retriever.",
        ]
        self.assertEqual(
            self.processor.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
