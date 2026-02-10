from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_base
from comfy_api.latest import ComfyExtension, io


def _ensure_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    # Expected MASK tensor shape in ComfyUI: [B, H, W]
    if mask.ndim == 2:
        return mask.unsqueeze(0)
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        # Some nodes may provide [B, 1, H, W]
        return mask[:, 0]
    raise ValueError(f"Unsupported MASK tensor rank for EliGen: {mask.ndim}")


@dataclass
class _EliGenRuntimeContext:
    base_txt_tokens: int
    image_tokens: int
    entity_token_ranges: list[tuple[int, int]]
    entity_assignments: torch.Tensor  # [B, image_tokens], long, -1 = background
    apply_per_batch: torch.Tensor  # [B], bool
    strength: float


class QwenImageEliGenPatch:
    def __init__(
        self,
        entity_prompt_embeds: list[torch.Tensor],
        entity_masks: torch.Tensor,
        txt_norm: torch.nn.Module,
        txt_in: torch.nn.Module,
        enable_on_negative: bool = False,
        strength: float = 1.0,
    ):
        # raw encoder outputs (Qwen TE width 3584), projected later into model width 3072
        self.entity_prompt_embeds = [e.detach().cpu() for e in entity_prompt_embeds]
        self.entity_masks = entity_masks.detach().float().cpu()
        self.txt_norm = txt_norm
        self.txt_in = txt_in
        self.enable_on_negative = bool(enable_on_negative)
        self.strength = float(strength)

    def _entity_prompt_embeds_for(self, txt: torch.Tensor):
        embeds = []
        ranges = []
        start = txt.shape[1]

        for emb in self.entity_prompt_embeds:
            emb_t = emb.to(device=txt.device, dtype=txt.dtype)
            if emb_t.shape[0] != txt.shape[0]:
                emb_t = emb_t.repeat(txt.shape[0], 1, 1)
            emb_t = self.txt_in(self.txt_norm(emb_t))
            end = start + emb_t.shape[1]
            ranges.append((start, end))
            start = end
            embeds.append(emb_t)

        total_tokens = start - txt.shape[1]
        return embeds, ranges, total_tokens

    def _entity_assignment_for_img_tokens(self, image_tokens: int, batch_size: int, device: torch.device):
        if image_tokens <= 0:
            return torch.full((batch_size, 0), -1, dtype=torch.long, device=device)

        side = int(round(image_tokens ** 0.5))
        if side * side > image_tokens:
            side -= 1
        if side <= 0:
            side = 1

        token_h, token_w = side, max(1, image_tokens // side)
        if token_h * token_w > image_tokens:
            token_w = max(1, image_tokens // token_h)

        masks = self.entity_masks.to(device=device)
        masks = F.interpolate(masks.unsqueeze(1), size=(token_h, token_w), mode="nearest").squeeze(1)
        max_vals, max_idx = masks.max(dim=0)
        assignment_hw = torch.where(max_vals > 0.5, max_idx, torch.full_like(max_idx, -1)).reshape(1, -1)

        if assignment_hw.shape[1] < image_tokens:
            pad = torch.full((1, image_tokens - assignment_hw.shape[1]), -1, dtype=assignment_hw.dtype, device=assignment_hw.device)
            assignment = torch.cat([assignment_hw, pad], dim=1)
        else:
            assignment = assignment_hw[:, :image_tokens]

        return assignment.repeat(batch_size, 1)

    def _build_runtime_context(self, args: dict, txt_ext: torch.Tensor, entity_token_ranges):
        img = args["img"]
        to = args["transformer_options"]

        base_txt_tokens = txt_ext.shape[1] - sum((r[1] - r[0]) for r in entity_token_ranges)
        image_tokens = img.shape[1]
        entity_assignments = self._entity_assignment_for_img_tokens(image_tokens, img.shape[0], img.device)

        apply_per_batch = torch.ones((img.shape[0],), dtype=torch.bool, device=img.device)
        if not self.enable_on_negative:
            cond_or_uncond = to.get("cond_or_uncond", None)
            if isinstance(cond_or_uncond, list) and len(cond_or_uncond) == img.shape[0]:
                # Comfy sampler convention: 0 => positive, 1 => negative
                apply_per_batch = torch.tensor([v == 0 for v in cond_or_uncond], dtype=torch.bool, device=img.device)

        return _EliGenRuntimeContext(
            base_txt_tokens=base_txt_tokens,
            image_tokens=image_tokens,
            entity_token_ranges=entity_token_ranges,
            entity_assignments=entity_assignments,
            apply_per_batch=apply_per_batch,
            strength=self.strength,
        )

    def _attention_override(self, func, *attn_args, **attn_kwargs):
        transformer_options = attn_kwargs.get("transformer_options", {})
        ctx = transformer_options.get("_qwen_eligen_ctx", None)
        if ctx is None:
            return func(*attn_args, **attn_kwargs)

        q = attn_args[0]
        if q.ndim != 4:
            return func(*attn_args, **attn_kwargs)

        b, _, seq_q, _ = q.shape
        total_txt = ctx.base_txt_tokens + sum((end - start) for start, end in ctx.entity_token_ranges)
        if seq_q < total_txt or total_txt <= 0:
            return func(*attn_args, **attn_kwargs)

        total_len = seq_q
        img_start = total_txt
        neg = -80.0 * max(0.0, min(ctx.strength, 10.0))

        eligen_bias = torch.zeros((b, 1, total_len, total_len), device=q.device, dtype=q.dtype)
        img_assign = ctx.entity_assignments.to(device=q.device)

        for entity_id, (tok_start, tok_end) in enumerate(ctx.entity_token_ranges):
            disallow = (img_assign != entity_id).unsqueeze(1).unsqueeze(-1)
            eligen_bias[:, :, img_start:, tok_start:tok_end] = torch.where(
                disallow,
                torch.full((1,), neg, device=q.device, dtype=q.dtype),
                torch.zeros((1,), device=q.device, dtype=q.dtype),
            )

        # apply only on selected batch rows (positive by default)
        row_mask = ctx.apply_per_batch.to(device=q.device, dtype=q.dtype).view(b, 1, 1, 1)
        eligen_bias = eligen_bias * row_mask

        mask = attn_args[4] if len(attn_args) > 4 else None
        new_mask = eligen_bias if mask is None else (mask + eligen_bias)

        attn_args = list(attn_args)
        if len(attn_args) > 4:
            attn_args[4] = new_mask
        else:
            attn_args.append(new_mask)

        return func(*attn_args, **attn_kwargs)

    def __call__(self, args, extra_args):
        txt = args["txt"]
        if len(self.entity_prompt_embeds) == 0:
            return extra_args["original_block"](args)

        entity_embeds, entity_ranges, entity_total_tokens = self._entity_prompt_embeds_for(txt)
        if entity_total_tokens == 0:
            return extra_args["original_block"](args)

        txt_ext = torch.cat([txt] + entity_embeds, dim=1)
        args_mod = args.copy()
        args_mod["txt"] = txt_ext

        to = args_mod["transformer_options"]
        prev_override = to.get("optimized_attention_override", None)
        prev_ctx = to.get("_qwen_eligen_ctx", None)

        to["optimized_attention_override"] = self._attention_override
        to["_qwen_eligen_ctx"] = self._build_runtime_context(args_mod, txt_ext, entity_ranges)

        try:
            out = extra_args["original_block"](args_mod)
        finally:
            if prev_override is None:
                to.pop("optimized_attention_override", None)
            else:
                to["optimized_attention_override"] = prev_override

            if prev_ctx is None:
                to.pop("_qwen_eligen_ctx", None)
            else:
                to["_qwen_eligen_ctx"] = prev_ctx

        out["txt"] = out["txt"][:, :txt.shape[1], :]
        return out

    def to(self, device_or_dtype):
        return self


class ApplyQwenImageEliGen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ApplyQwenImageEliGen",
            display_name="Apply Qwen Image EliGen",
            search_aliases=["eligen", "qwen eligen", "entity control"],
            category="advanced/loaders/qwen",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Bool.Input("eligen_enable_on_negative", default=False),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),

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
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        eligen_enable_on_negative,
        strength,
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
        **kwargs,
    ) -> io.NodeOutput:
        if not isinstance(model.model, comfy.model_base.QwenImage):
            raise ValueError("ApplyQwenImageEliGen only supports Qwen Image models.")

        raw_pairs = [
            (entity_prompt_1, mask_1), (entity_prompt_2, mask_2), (entity_prompt_3, mask_3), (entity_prompt_4, mask_4), (entity_prompt_5, mask_5),
            (entity_prompt_6, mask_6), (entity_prompt_7, mask_7), (entity_prompt_8, mask_8), (entity_prompt_9, mask_9), (entity_prompt_10, mask_10),
        ]

        prompts = []
        masks = []
        for idx, (p, m) in enumerate(raw_pairs, start=1):
            p = (p or "").strip()
            if p == "" and m is None:
                continue
            if p == "" and m is not None:
                raise ValueError(f"mask_{idx} is connected but entity_prompt_{idx} is empty.")
            if p != "" and m is None:
                raise ValueError(f"entity_prompt_{idx} is set but mask_{idx} is not connected.")
            mm = _ensure_mask_tensor(m)
            prompts.append(p)
            masks.append(mm[0:1])

        if len(prompts) == 0:
            raise ValueError("At least one entity prompt/mask pair is required.")

        masks_t = torch.cat(masks, dim=0)

        diffusion_model = model.get_model_object("diffusion_model")
        txt_norm = diffusion_model.txt_norm
        txt_in = diffusion_model.txt_in

        embeds = []
        for p in prompts:
            tokens = clip.tokenize(p)
            enc = clip.encode_from_tokens(tokens, return_dict=True)
            embeds.append(enc["cond"])

        patch = QwenImageEliGenPatch(
            embeds,
            masks_t,
            txt_norm=txt_norm,
            txt_in=txt_in,
            enable_on_negative=eligen_enable_on_negative,
            strength=strength,
        )

        model_patched = model.clone()
        block_count = len(model_patched.get_model_object("diffusion_model.transformer_blocks"))
        for i in range(block_count):
            model_patched.set_model_patch_replace(patch, "dit", "double_block", i)

        return io.NodeOutput(model_patched)


class QwenEliGenExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ApplyQwenImageEliGen,
        ]


async def comfy_entrypoint() -> QwenEliGenExtension:
    return QwenEliGenExtension()

