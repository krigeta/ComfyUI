import torch
from typing_extensions import override

import comfy.model_base
import node_helpers
from comfy_api.latest import ComfyExtension, io


def _ensure_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    # Expected MASK tensor shape in ComfyUI: [B, H, W]
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        return mask[:, 0]
    raise ValueError(f"Unsupported MASK tensor rank for EliGen: {mask.ndim}")


def _normalize_mask(mask: torch.Tensor, target_h: int | None = None, target_w: int | None = None) -> torch.Tensor:
    m = _ensure_mask_tensor(mask).float()
    if target_h is not None and target_w is not None and (m.shape[-2] != target_h or m.shape[-1] != target_w):
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(target_h, target_w), mode="nearest").squeeze(1)
    m = (m > 0).float()
    if torch.sum(m) <= 0:
        raise ValueError("Entity mask has no active pixels after normalization.")
    return m


def _encode_entity_prompt_qwen(clip, prompt: str):
    # Match QwenImage tokenizer template behavior
    tokens = clip.tokenize(prompt)
    enc = clip.encode_from_tokens(tokens, return_dict=True)
    prompt_emb = enc["cond"]
    prompt_mask = enc.get("attention_mask", None)
    if prompt_mask is None:
        prompt_mask = torch.ones((prompt_emb.shape[0], prompt_emb.shape[1]), dtype=torch.long, device=prompt_emb.device)
    return prompt_emb, prompt_mask


class ApplyQwenImageEliGen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ApplyQwenImageEliGenCore",
            display_name="Apply Qwen Image EliGen",
            search_aliases=["eligen", "qwen eligen", "entity control"],
            category="advanced/loaders/qwen",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Boolean.Input("eligen_enable_on_negative", default=False),

                io.String.Input("entity_prompt_1", default="", multiline=True, optional=True),
                io.Mask.Input("mask_1", optional=True),
                io.String.Input("entity_prompt_2", default="", multiline=True, optional=True),
                io.Mask.Input("mask_2", optional=True),
                io.String.Input("entity_prompt_3", default="", multiline=True, optional=True),
                io.Mask.Input("mask_3", optional=True),
                io.String.Input("entity_prompt_4", default="", multiline=True, optional=True),
                io.Mask.Input("mask_4", optional=True),
                io.String.Input("entity_prompt_5", default="", multiline=True, optional=True),
                io.Mask.Input("mask_5", optional=True),
                io.String.Input("entity_prompt_6", default="", multiline=True, optional=True),
                io.Mask.Input("mask_6", optional=True),
                io.String.Input("entity_prompt_7", default="", multiline=True, optional=True),
                io.Mask.Input("mask_7", optional=True),
                io.String.Input("entity_prompt_8", default="", multiline=True, optional=True),
                io.Mask.Input("mask_8", optional=True),
                io.String.Input("entity_prompt_9", default="", multiline=True, optional=True),
                io.Mask.Input("mask_9", optional=True),
                io.String.Input("entity_prompt_10", default="", multiline=True, optional=True),
                io.Mask.Input("mask_10", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Model.Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        model,
        clip,
        eligen_enable_on_negative,
        entity_prompt_1="", mask_1=None,
        entity_prompt_2="", mask_2=None,
        entity_prompt_3="", mask_3=None,
        entity_prompt_4="", mask_4=None,
        entity_prompt_5="", mask_5=None,
        entity_prompt_6="", mask_6=None,
        entity_prompt_7="", mask_7=None,
        entity_prompt_8="", mask_8=None,
        entity_prompt_9="", mask_9=None,
        entity_prompt_10="", mask_10=None,
    ) -> io.NodeOutput:
        if not isinstance(model.model, comfy.model_base.QwenImage):
            raise ValueError("ApplyQwenImageEliGen only supports Qwen Image models.")

        raw_pairs = [
            (entity_prompt_1, mask_1), (entity_prompt_2, mask_2), (entity_prompt_3, mask_3), (entity_prompt_4, mask_4), (entity_prompt_5, mask_5),
            (entity_prompt_6, mask_6), (entity_prompt_7, mask_7), (entity_prompt_8, mask_8), (entity_prompt_9, mask_9), (entity_prompt_10, mask_10),
        ]

        target_h = None
        target_w = None
        first_mask = None
        for p, m in raw_pairs:
            p = (p or "").strip()
            if p != "" and m is not None:
                first_mask = _ensure_mask_tensor(m)
                break

        if first_mask is not None:
            target_h, target_w = first_mask.shape[-2], first_mask.shape[-1]

        entity_prompt_emb = []
        entity_prompt_emb_mask = []
        masks = []

        for idx, (p, m) in enumerate(raw_pairs, start=1):
            p = (p or "").strip()
            if p == "" and m is None:
                continue
            if p == "" and m is not None:
                raise ValueError(f"mask_{idx} is connected but entity_prompt_{idx} is empty.")
            if p != "" and m is None:
                raise ValueError(f"entity_prompt_{idx} is set but mask_{idx} is not connected.")

            emb, emb_mask = _encode_entity_prompt_qwen(clip, p)
            entity_prompt_emb.append(emb)
            entity_prompt_emb_mask.append(emb_mask)
            mm = _normalize_mask(m, target_h=target_h, target_w=target_w)
            masks.append(mm[0:1])

        if len(entity_prompt_emb) == 0:
            raise ValueError("At least one entity prompt/mask pair is required.")

        entity_masks = torch.cat(masks, dim=0).unsqueeze(0).unsqueeze(2)

        positive = node_helpers.conditioning_set_values(
            positive,
            {
                "eligen_entity_prompt_emb": entity_prompt_emb,
                "eligen_entity_prompt_emb_mask": entity_prompt_emb_mask,
                "eligen_entity_masks": entity_masks,
            },
            append=False,
        )

        if eligen_enable_on_negative:
            negative = node_helpers.conditioning_set_values(
                negative,
                {
                    "eligen_entity_prompt_emb": entity_prompt_emb,
                    "eligen_entity_prompt_emb_mask": entity_prompt_emb_mask,
                    "eligen_entity_masks": entity_masks,
                },
                append=False,
            )

        return io.NodeOutput(positive, negative, model)


class QwenEliGenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ApplyQwenImageEliGen,
        ]


async def comfy_entrypoint() -> QwenEliGenExtension:
    return QwenEliGenExtension()

