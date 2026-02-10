from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing_extensions import override

import comfy.model_base
from comfy_api.latest import ComfyExtension, io


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


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
    entity_assignments: torch.Tensor  # [B, image_tokens], long, -1 means background
    strength: float


class QwenImageEliGenPatch:
    def __init__(
        self,
        entity_prompt_embeds: list[torch.Tensor],
        entity_masks: torch.Tensor,
        txt_norm: torch.nn.Module,
        txt_in: torch.nn.Module,
        strength: float = 1.0,
    ):
        # Store raw CLIP/Qwen text embeddings (3584 for Qwen Image TE), then
        # project to model inner dim (3072) inside the patch using model txt_norm/txt_in.
        self.entity_prompt_embeds = [e.detach().cpu() for e in entity_prompt_embeds]
        self.entity_masks = entity_masks.detach().float().cpu()
        self.txt_norm = txt_norm
        self.txt_in = txt_in
        self.strength = float(strength)

    def _entity_prompt_embeds_for(self, txt: torch.Tensor) -> tuple[list[torch.Tensor], list[tuple[int, int]], int]:
        embeds = []
        ranges = []
        start = txt.shape[1]
        for emb in self.entity_prompt_embeds:
            emb_t = emb.to(device=txt.device, dtype=txt.dtype)
            if emb_t.shape[0] != txt.shape[0]:
                emb_t = emb_t.repeat(txt.shape[0], 1, 1)
            # Project from text-encoder hidden dim to QwenImage transformer dim.
            emb_t = self.txt_in(self.txt_norm(emb_t))
            end = start + emb_t.shape[1]
            ranges.append((start, end))
            start = end
            embeds.append(emb_t)
        total_tokens = start - txt.shape[1]
        return embeds, ranges, total_tokens

    def _entity_assignment_for(self, x: torch.Tensor, image_tokens: int, batch_size: int, device: torch.device) -> torch.Tensor:
        # x for Qwen-Image is [B, C, T, H, W]
        if x.ndim != 5:
            return torch.full((batch_size, image_tokens), -1, dtype=torch.long, device=device)

        _, _, t, h, w = x.shape
        token_h = (h + 1) // 2
        token_w = (w + 1) // 2

        masks = self.entity_masks.to(device=device)
        masks = F.interpolate(masks.unsqueeze(1), size=(token_h, token_w), mode="nearest").squeeze(1)

        # resolve overlaps by argmax, background = all zeros
        max_vals, max_idx = masks.max(dim=0)  # [token_h, token_w]
        assignment_hw = torch.where(max_vals > 0.5, max_idx, torch.full_like(max_idx, -1))
        assignment_hw = assignment_hw.reshape(1, -1)  # [1, token_h * token_w]

        # Qwen image token order uses T first, then H/W in each frame
        assignment = assignment_hw.repeat(1, t).reshape(1, -1)

        if assignment.shape[1] != image_tokens:
            # safety fallback when shape assumptions drift
            return torch.full((batch_size, image_tokens), -1, dtype=torch.long, device=device)

        return assignment.repeat(batch_size, 1)

    def _build_runtime_context(self, args: dict, txt_ext: torch.Tensor, entity_token_ranges: list[tuple[int, int]]) -> _EliGenRuntimeContext:
        img = args["img"]
        x = args["x"]
        base_txt_tokens = txt_ext.shape[1] - sum((r[1] - r[0]) for r in entity_token_ranges)
        image_tokens = img.shape[1]
        entity_assignments = self._entity_assignment_for(x, image_tokens, img.shape[0], img.device)
        return _EliGenRuntimeContext(
            base_txt_tokens=base_txt_tokens,
            image_tokens=image_tokens,
            entity_token_ranges=entity_token_ranges,
            entity_assignments=entity_assignments,
            strength=self.strength,
        )

    def _attention_override(self, func, *attn_args, **attn_kwargs):
        transformer_options = attn_kwargs.get("transformer_options", {})
        ctx: _EliGenRuntimeContext | None = transformer_options.get("_qwen_eligen_ctx", None)
        if ctx is None:
            return func(*attn_args, **attn_kwargs)

        q = attn_args[0]
        if q.ndim != 4:
            return func(*attn_args, **attn_kwargs)

        b, _, seq_q, _ = q.shape
        total_txt = ctx.base_txt_tokens + sum((end - start) for start, end in ctx.entity_token_ranges)
        if seq_q < total_txt:
            return func(*attn_args, **attn_kwargs)

        total_len = seq_q
        img_start = total_txt
        neg = -torch.finfo(q.dtype).max * max(0.0, min(ctx.strength, 10.0))

        eligen_bias = torch.zeros((b, 1, total_len, total_len), device=q.device, dtype=q.dtype)
        img_assign = ctx.entity_assignments.to(device=q.device)

        for entity_id, (tok_start, tok_end) in enumerate(ctx.entity_token_ranges):
            disallow = (img_assign != entity_id).unsqueeze(1).unsqueeze(-1)  # [B,1,img_tokens,1]
            eligen_bias[:, :, img_start:, tok_start:tok_end] = torch.where(
                disallow,
                torch.full((1,), neg, device=q.device, dtype=q.dtype),
                torch.zeros((1,), device=q.device, dtype=q.dtype),
            )

        mask = attn_args[4] if len(attn_args) > 4 else None
        if mask is None:
            new_mask = eligen_bias
        else:
            new_mask = mask + eligen_bias

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

        # Keep original text-token length expected by the main model path.
        out["txt"] = out["txt"][:, :txt.shape[1], :]
        return out

    def to(self, device_or_dtype):
        # Called by ComfyUI patch loader. Keep tensors on CPU; move dynamically at runtime.
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
                io.String.Input(
                    "entity_prompts",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip="One entity prompt per line, in the same order as mask_1 ... mask_10.",
                ),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Mask.Input("mask_1", optional=True),
                io.Mask.Input("mask_2", optional=True),
                io.Mask.Input("mask_3", optional=True),
                io.Mask.Input("mask_4", optional=True),
                io.Mask.Input("mask_5", optional=True),
                io.Mask.Input("mask_6", optional=True),
                io.Mask.Input("mask_7", optional=True),
                io.Mask.Input("mask_8", optional=True),
                io.Mask.Input("mask_9", optional=True),
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
        entity_prompts,
        strength,
        mask_1=None,
        mask_2=None,
        mask_3=None,
        mask_4=None,
        mask_5=None,
        mask_6=None,
        mask_7=None,
        mask_8=None,
        mask_9=None,
        mask_10=None,
    ) -> io.NodeOutput:
        if not isinstance(model.model, comfy.model_base.QwenImage):
            raise ValueError("ApplyQwenImageEliGen only supports Qwen Image models.")

        prompts = _split_lines(entity_prompts)
        if len(prompts) == 0:
            return io.NodeOutput(model)

        raw_masks = [mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8, mask_9, mask_10]
        masks_list = []
        for m in raw_masks:
            if m is not None:
                mm = _ensure_mask_tensor(m)
                # take first mask in case an upstream node sends batch; one input socket = one entity
                masks_list.append(mm[0:1])

        if len(masks_list) == 0:
            raise ValueError("At least one entity mask (mask_1 ... mask_10) is required.")

        if len(prompts) > len(masks_list):
            raise ValueError(f"Entity prompts count ({len(prompts)}) exceeds provided masks ({len(masks_list)}).")

        # Use as many pairs as available, up to 10.
        pair_count = min(len(prompts), len(masks_list), 10)
        prompts = prompts[:pair_count]
        masks = torch.cat(masks_list[:pair_count], dim=0)

        diffusion_model = model.get_model_object("diffusion_model")
        txt_norm = diffusion_model.txt_norm
        txt_in = diffusion_model.txt_in

        embeds = []
        for p in prompts:
            tokens = clip.tokenize(p)
            enc = clip.encode_from_tokens(tokens, return_dict=True)
            embeds.append(enc["cond"])

        patch = QwenImageEliGenPatch(embeds, masks, txt_norm=txt_norm, txt_in=txt_in, strength=strength)

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

