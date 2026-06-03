"""
Qwen2.5-Omni frontend implementation.

Adapter for `Qwen/Qwen2.5-Omni-7B` (a unified omni-modal model that listens to
audio and emits text). Key characteristics:

- Uses `Qwen2_5OmniForConditionalGeneration` / `Qwen2_5OmniProcessor` from
  transformers (note: underscore in the class name).
- Single-GPU practical: weights are ~16GB (vs ~60GB for Qwen3-Omni 30B).
- Same multimodal preprocessing via `qwen_omni_utils.process_mm_info`.

Constraints (intentionally narrow):
- audio + text input only
- text output only
- no video-audio path (`use_audio_in_video=False`)

Required environment:
- transformers >= 4.45 (Qwen2.5-Omni support landed in 4.45) OR install from
  source via `pip install git+https://github.com/huggingface/transformers`.
- qwen_omni_utils (multimodal preprocessing helpers).
- torch with CUDA (the model needs a GPU).
"""

from typing import Any

from audio_agent.core.errors import FrontendError
from audio_agent.frontend.model_frontend import (
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)
from audio_agent.utils.model_downloader import DEFAULT_QWEN25_OMNI_PATH

DEFAULT_QWEN25_OMNI_MODEL_PATH = DEFAULT_QWEN25_OMNI_PATH


class Qwen25OmniFrontend(BaseModelFrontend):
    """Qwen2.5-Omni-7B frontend adapter."""

    def __init__(
        self,
        model_path: str = DEFAULT_QWEN25_OMNI_MODEL_PATH,
        model_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str = "flash_attention_2",
        generation_kwargs: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_path = model_path
        self.use_audio_in_video = False
        self.model_dtype = model_dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_config=model_config, max_retries=max_retries)

    @property
    def name(self) -> str:
        return "qwen25_omni_frontend"

    @property
    def input_format(self) -> FrontendInputFormat:
        return FrontendInputFormat.LOCAL_MULTIMODAL

    def initialize_model(self) -> dict[str, Any]:
        """Load Qwen2.5-Omni model + processor."""
        try:
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as e:
            raise FrontendError(
                "Missing Qwen2.5-Omni dependencies",
                details={
                    "required_packages": [
                        "transformers (>=4.45 or git main)",
                        "qwen_omni_utils",
                        "torch",
                        "soundfile",
                    ],
                    "hint": (
                        "Install with: pip install -U transformers qwen_omni_utils soundfile. "
                        "If the symbol Qwen2_5OmniForConditionalGeneration is missing, install "
                        "transformers from git: pip install git+https://github.com/huggingface/transformers"
                    ),
                },
            ) from e

        try:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=self.model_dtype,
                device_map=self.device_map,
                attn_implementation=self.attn_implementation,
            )
            processor = Qwen2_5OmniProcessor.from_pretrained(self.model_path)
        except Exception as e:
            raise FrontendError(
                f"Failed to initialize Qwen2.5-Omni model/processor: {type(e).__name__}: {e}",
                details={"model_path": self.model_path},
            ) from e

        return {
            "model": model,
            "processor": processor,
        }

    def call_model(self, model_input: UnifiedFrontendInput) -> str:
        """Run Qwen2.5-Omni generation and return raw text output."""
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as e:
            raise FrontendError(
                "Missing qwen_omni_utils dependency",
                details={"required_package": "qwen_omni_utils"},
            ) from e

        model = self.model_handle["model"]
        processor = self.model_handle["processor"]
        conversation = model_input.messages

        try:
            text = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audios, images, videos = process_mm_info(
                conversation,
                use_audio_in_video=self.use_audio_in_video,
            )
            if images or videos:
                raise FrontendError(
                    "Qwen25OmniFrontend only supports audio+text input",
                    details={
                        "images_count": len(images) if images is not None else 0,
                        "videos_count": len(videos) if videos is not None else 0,
                    },
                )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
            inputs = inputs.to(model.device)
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is not None:
                inputs = inputs.to(model_dtype)

            # Qwen2.5-Omni's generate() supports the same thinker_return_dict_in_generate
            # contract as Qwen3-Omni. Disallow speech-output params.
            generate_kwargs: dict[str, Any] = {
                "thinker_return_dict_in_generate": True,
                "use_audio_in_video": self.use_audio_in_video,
            }
            if "speaker" in self.generation_kwargs:
                raise FrontendError(
                    "speaker generation option is disabled for text-only frontend mode"
                )
            generate_kwargs.update(self.generation_kwargs)

            text_ids, _audio = model.generate(**inputs, **generate_kwargs)
            response_texts = processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as e:
            raise FrontendError(
                f"Qwen2.5-Omni inference failed: {type(e).__name__}: {e}",
                details={"model_path": self.model_path},
            ) from e

        if not response_texts:
            raise FrontendError("Qwen2.5-Omni returned empty response list")

        raw_response = response_texts[0].strip()
        if not raw_response:
            raise FrontendError("Qwen2.5-Omni returned empty response text")

        return raw_response
