"""Arc2Face helpers: text-encoder wrapper, identity-embedding projection and model loaders.

The ``CLIPTextModelWrapper`` and ``project_face_embs`` below are adapted from the official
Arc2Face repository (https://github.com/foivospar/Arc2Face), MIT License,
Copyright (c) 2024 Foivos Paraperas Papantoniou.
The Arc2Face *model weights* are distributed separately by the authors under their own
(non-commercial) terms; see the README.
"""
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import CLIPTextModel
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

SD15_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class CLIPTextModelWrapper(CLIPTextModel):
    """CLIP text encoder that can take precomputed token embeddings (``input_token_embs``)
    or return the token embeddings of ``input_ids`` (``return_token_embs=True``)."""

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_token_embs: Optional[torch.Tensor] = None,
        return_token_embs: Optional[bool] = False,
    ) -> Union[Tuple, torch.Tensor, BaseModelOutputWithPooling]:
        if return_token_embs:
            return self.text_model.embeddings.token_embedding(input_ids)

        cfg = self.text_model.config
        output_attentions = output_attentions if output_attentions is not None else cfg.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else cfg.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else cfg.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.text_model.embeddings(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=input_token_embs
        )
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        if attention_mask is not None and not self._use_flash_attention_2:
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.text_model.final_layer_norm(encoder_outputs[0])

        batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
        ids = input_ids.to(dtype=torch.int, device=last_hidden_state.device)
        if self.text_model.eos_token_id == 2:
            pooled_output = last_hidden_state[batch_idx, ids.argmax(dim=-1)]
        else:
            pooled_output = last_hidden_state[
                batch_idx, (ids == self.text_model.eos_token_id).int().argmax(dim=-1)
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@torch.no_grad()
def project_face_embs(pipeline, face_embs: torch.Tensor) -> torch.Tensor:
    """Project (N, 512) ArcFace embeddings to Arc2Face prompt embeddings."""
    arcface_token_id = pipeline.tokenizer.encode("id", add_special_tokens=False)[0]
    input_ids = pipeline.tokenizer(
        "photo of a id person",
        truncation=True,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(pipeline.device)
    input_ids = input_ids.repeat(len(face_embs), 1)
    face_embs_padded = F.pad(
        face_embs, (0, pipeline.text_encoder.config.hidden_size - 512), "constant", 0
    )
    token_embs = pipeline.text_encoder(input_ids, return_token_embs=True)
    token_embs[input_ids == arcface_token_id] = face_embs_padded.to(
        dtype=token_embs.dtype, device=token_embs.device
    )
    return pipeline.text_encoder(
        input_ids=input_ids, input_token_embs=token_embs, attention_mask=None
    )[0]


def build_arc2face_pipeline(arc2face_models_dir: str, device: str, dtype: torch.dtype):
    """Stable Diffusion 1.5 pipeline with the Arc2Face encoder and UNet."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel

    encoder = CLIPTextModelWrapper.from_pretrained(
        arc2face_models_dir, subfolder="encoder", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        arc2face_models_dir, subfolder="arc2face", torch_dtype=dtype
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        SD15_REPO, text_encoder=encoder, unet=unet, torch_dtype=dtype, safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline.to(device)


def build_face_analyzer(insightface_root: str, device_id: int = 0):
    """InsightFace (antelopev2) analyzer used to re-extract ArcFace embeddings from images."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root=insightface_root,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"device_id": device_id}, {}],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app
