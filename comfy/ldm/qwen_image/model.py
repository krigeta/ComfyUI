# https://github.com/QwenLM/Qwen-Image (Apache 2.0)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import repeat

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
import comfy.ldm.common_dit
import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope1
from einops import rearrange

class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, operations=operations))
        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, bias=bias, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb(x, freqs_cis):
    if x.shape[1] == 0:
        return x

    t_ = x.reshape(*x.shape[:-1], -1, 1, 2)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.reshape(*x.shape)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, use_additional_t_cond=False, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.use_additional_t_cond = use_additional_t_cond
        if self.use_additional_t_cond:
            self.addition_t_embedding = operations.Embedding(2, embedding_dim, device=device, dtype=dtype)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))

        if self.use_additional_t_cond:
            if addition_t_cond is None:
                addition_t_cond = torch.zeros((timesteps_emb.shape[0]), device=timesteps_emb.device, dtype=torch.long)
            timesteps_emb += self.addition_t_embedding(addition_t_cond, out_dtype=timesteps_emb.dtype)

        return timesteps_emb


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Text stream projections
        self.add_q_proj = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Output projections
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype, device=device),
            nn.Dropout(dropout)
        ])
        self.to_add_out = operations.Linear(self.inner_dim, self.out_context_dim, bias=out_bias, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        seq_img = hidden_states.shape[1]
        seq_txt = encoder_hidden_states.shape[1]

        # Project and reshape to BHND format (batch, heads, seq, dim)
        img_query = self.to_q(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_key = self.to_k(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2).contiguous()
        img_value = self.to_v(hidden_states).view(batch_size, seq_img, self.heads, -1).transpose(1, 2)

        txt_query = self.add_q_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_key = self.add_k_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2).contiguous()
        txt_value = self.add_v_proj(encoder_hidden_states).view(batch_size, seq_txt, self.heads, -1).transpose(1, 2)

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        joint_query = torch.cat([txt_query, img_query], dim=2)
        joint_key = torch.cat([txt_key, img_key], dim=2)
        joint_value = torch.cat([txt_value, img_value], dim=2)

        joint_query = apply_rope1(joint_query, image_rotary_emb)
        joint_key = apply_rope1(joint_key, image_rotary_emb)

        if encoder_hidden_states_mask is not None:
            # Two supported mask forms:
            # 1) text-only keep mask [B, seq_txt] -> expand to joint [B, 1, seq_txt + seq_img]
            # 2) full joint attention mask [B, 1, seq_total, seq_total] (EliGen path)
            if encoder_hidden_states_mask.ndim == 2:
                attn_mask = torch.zeros((batch_size, 1, seq_txt + seq_img), dtype=hidden_states.dtype, device=hidden_states.device)
                attn_mask[:, 0, :seq_txt] = encoder_hidden_states_mask
            else:
                attn_mask = encoder_hidden_states_mask
        else:
            attn_mask = None

        joint_hidden_states = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads,
                                                         attn_mask, transformer_options=transformer_options,
                                                         skip_reshape=True)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _apply_gate(self, x, y, gate, timestep_zero_index=None):
        if timestep_zero_index is not None:
            return y + torch.cat((x[:, :timestep_zero_index] * gate[0], x[:, timestep_zero_index:] * gate[1]), dim=1)
        else:
            return torch.addcmul(y, gate, x)

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor, timestep_zero_index=None) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = torch.chunk(mod_params, 3, dim=-1)
        if timestep_zero_index is not None:
            actual_batch = shift.size(0) // 2
            shift, shift_0 = shift[:actual_batch], shift[actual_batch:]
            scale, scale_0 = scale[:actual_batch], scale[actual_batch:]
            gate, gate_0 = gate[:actual_batch], gate[actual_batch:]
            reg = torch.addcmul(shift.unsqueeze(1), x[:, :timestep_zero_index], 1 + scale.unsqueeze(1))
            zero = torch.addcmul(shift_0.unsqueeze(1), x[:, timestep_zero_index:], 1 + scale_0.unsqueeze(1))
            return torch.cat((reg, zero), dim=1), (gate.unsqueeze(1), gate_0.unsqueeze(1))
        else:
            return torch.addcmul(shift.unsqueeze(1), x, 1 + scale.unsqueeze(1)), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        timestep_zero_index=None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)

        if timestep_zero_index is not None:
            temb = temb.chunk(2, dim=0)[0]

        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1, timestep_zero_index)
        del img_mod1
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        del txt_mod1

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )
        del img_modulated
        del txt_modulated

        hidden_states = self._apply_gate(img_attn_output, hidden_states, img_gate1, timestep_zero_index)
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        del img_attn_output
        del txt_attn_output
        del img_gate1
        del txt_gate1

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, timestep_zero_index)
        hidden_states = self._apply_gate(self.img_mlp(img_modulated2), hidden_states, img_gate2, timestep_zero_index)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = torch.addcmul(encoder_hidden_states, txt_gate2, self.txt_mlp(txt_modulated2))

        return encoder_hidden_states, hidden_states


class LastLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=False,
        eps=1e-6,
        bias=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = torch.addcmul(shift[:, None, :], self.norm(x), (1 + scale)[:, None, :])
        return x


class QwenImageTransformer2DModel(nn.Module):
    LATENT_TO_PIXEL_RATIO = 8
    PATCH_TO_LATENT_RATIO = 2
    PATCH_TO_PIXEL_RATIO = LATENT_TO_PIXEL_RATIO * PATCH_TO_LATENT_RATIO

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        default_ref_method="index",
        image_model=None,
        final_layer=True,
        use_additional_t_cond=False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.default_ref_method = default_ref_method

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            use_additional_t_cond=use_additional_t_cond,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                device=device,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        if self.default_ref_method == "index_timestep_zero":
            self.register_buffer("__index_timestep_zero__", torch.tensor([]))

        if final_layer:
            self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
            self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-3], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 3, 5, 1, 4, 6)
        hidden_states = hidden_states.reshape(orig_shape[0], orig_shape[-3] * (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        t_len = t
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device)

        if t_len > 1:
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).unsqueeze(1).unsqueeze(1)
        else:
            img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + index

        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1).unsqueeze(0) - (h_len // 2)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0) - (w_len // 2)
        return hidden_states, repeat(img_ids, "t h w c -> b (t h w) c", b=bs), orig_shape

    def process_entity_masks(
        self,
        latents,
        prompt_emb,
        prompt_emb_mask,
        entity_prompt_emb,
        entity_prompt_emb_mask,
        entity_masks,
        img_ids,
        base_image_seq_len,
        transformer_options={},
    ):
        batch_size = latents.shape[0]

        # normalize batch for entity prompt embeddings
        normalized_entity_emb = []
        for e in entity_prompt_emb:
            if e.shape[0] == 1 and prompt_emb.shape[0] > 1:
                e = e.repeat(prompt_emb.shape[0], 1, 1)
            normalized_entity_emb.append(self.txt_in(self.txt_norm(e)))

        all_prompt_emb = torch.cat(normalized_entity_emb + [prompt_emb], dim=1)

        # normalize batch for entity prompt attention masks
        normalized_entity_emb_mask = []
        for m in entity_prompt_emb_mask:
            if m.shape[0] == 1 and prompt_emb.shape[0] > 1:
                m = m.repeat(prompt_emb.shape[0], 1)
            normalized_entity_emb_mask.append(m)

        if prompt_emb_mask is not None and prompt_emb_mask.ndim == 2 and not torch.is_floating_point(prompt_emb_mask):
            global_seq_len = int(prompt_emb_mask[0].sum().item())
        else:
            global_seq_len = int(prompt_emb.shape[1])

        seq_lens = [int(m[0].sum().item()) for m in normalized_entity_emb_mask] + [global_seq_len]

        # rebuild text ids per-entity + global to align EliGen rotary indexing
        txt_start = round(max(((latents.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                              ((latents.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids_parts = []
        for s in seq_lens:
            ids = torch.arange(txt_start, txt_start + s, device=latents.device).reshape(1, -1, 1).repeat(batch_size, 1, 3)
            txt_ids_parts.append(ids)
        txt_ids = torch.cat(txt_ids_parts, dim=1)
        rope_ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(rope_ids).to(latents.dtype).contiguous()

        # normalize entity masks batch and shape: [B, N, 1, H, W]
        if entity_masks.ndim == 4:
            entity_masks = entity_masks.unsqueeze(2)
        if entity_masks.shape[0] == 1 and batch_size > 1:
            entity_masks = entity_masks.repeat(batch_size, 1, 1, 1, 1)
        entity_masks = entity_masks.to(device=latents.device, dtype=latents.dtype)

        # align masks to padded latent spatial size
        padded_h = latents.shape[-2]
        padded_w = latents.shape[-1]
        entity_masks = entity_masks.reshape(batch_size * entity_masks.shape[1], entity_masks.shape[2], entity_masks.shape[3], entity_masks.shape[4])
        entity_masks = torch.nn.functional.interpolate(entity_masks, size=(padded_h, padded_w), mode="nearest")
        entity_masks = entity_masks.reshape(batch_size, -1, 1, padded_h, padded_w)

        n_entity = entity_masks.shape[1]
        expanded_masks = entity_masks.repeat(1, 1, latents.shape[1], 1, 1)
        entity_mask_list = [expanded_masks[:, i] for i in range(n_entity)]
        global_mask = torch.ones_like(entity_mask_list[0])
        mask_groups = entity_mask_list + [global_mask]

        # deterministic patchification (2x2 on latent grid)
        patch_h = padded_h // self.PATCH_TO_LATENT_RATIO
        patch_w = padded_w // self.PATCH_TO_LATENT_RATIO
        single_patch_tokens = patch_h * patch_w
        total_image_seq = int(base_image_seq_len)

        patched_masks = []
        for m in mask_groups:
            patched = rearrange(
                m,
                "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=patch_h,
                W=patch_w,
                P=self.PATCH_TO_LATENT_RATIO,
                Q=self.PATCH_TO_LATENT_RATIO,
            )
            binary = torch.sum(patched, dim=-1) > 0
            if total_image_seq > single_patch_tokens:
                repeat_time = (total_image_seq + single_patch_tokens - 1) // single_patch_tokens
                binary = binary.repeat(1, repeat_time)[:, :total_image_seq]
            patched_masks.append(binary)

        # full attention matrix over [txt + img]
        total_txt = int(sum(seq_lens))
        total_seq = int(total_txt + total_image_seq)
        attn = torch.ones((batch_size, total_seq, total_seq), dtype=torch.bool, device=latents.device)

        cumsum = [0]
        for s in seq_lens:
            cumsum.append(cumsum[-1] + int(s))
        img_start = total_txt
        img_end = total_seq

        # prompt<->image restrictions
        for i in range(len(patched_masks)):
            p0, p1 = cumsum[i], cumsum[i + 1]
            image_mask = patched_masks[i].unsqueeze(1).repeat(1, max(1, p1 - p0), 1)
            attn[:, p0:p1, img_start:img_end] = image_mask
            attn[:, img_start:img_end, p0:p1] = image_mask.transpose(1, 2)

        # entity prompt isolation
        for i in range(len(seq_lens)):
            for j in range(len(seq_lens)):
                if i == j:
                    continue
                i0, i1 = cumsum[i], cumsum[i + 1]
                j0, j1 = cumsum[j], cumsum[j + 1]
                attn[:, i0:i1, j0:j1] = False

        attn = attn.float()
        attn[attn == 0] = float("-inf")
        attn[attn == 1] = 0

        # CFG-aware mask policy: keep negative unconstrained unless explicitly requested
        cond_or_uncond = transformer_options.get("cond_or_uncond", []) if transformer_options is not None else []
        if len(cond_or_uncond) == batch_size and 0 in cond_or_uncond and 1 in cond_or_uncond:
            standard = torch.zeros_like(attn)
            selected = []
            for i, cond_type in enumerate(cond_or_uncond):
                if cond_type == 0:
                    selected.append(attn[i:i + 1])
                else:
                    selected.append(standard[i:i + 1])
            attn = torch.cat(selected, dim=0)

        return all_prompt_emb, image_rotary_emb, attn.unsqueeze(1).to(dtype=latents.dtype)

    def forward(self, x, timestep, context, attention_mask=None, ref_latents=None, additional_t_cond=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, ref_latents, additional_t_cond, transformer_options, **kwargs)

    def _forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        ref_latents=None,
        additional_t_cond=None,
        transformer_options={},
        control=None,
        eligen_entity_prompt_emb=None,
        eligen_entity_prompt_emb_mask=None,
        eligen_entity_masks=None,
        **kwargs
    ):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        if encoder_hidden_states_mask is not None and not torch.is_floating_point(encoder_hidden_states_mask):
            encoder_hidden_states_mask = (encoder_hidden_states_mask - 1).to(x.dtype) * torch.finfo(x.dtype).max

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        timestep_zero_index = None
        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
            index_ref_method = (ref_method == "index") or (ref_method == "index_timestep_zero")
            negative_ref_method = ref_method == "negative_index"
            timestep_zero = ref_method == "index_timestep_zero"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                elif negative_ref_method:
                    index -= 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
            if timestep_zero:
                if index > 0:
                    timestep = torch.cat([timestep, timestep * 0], dim=0)
                    timestep_zero_index = num_embeds

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # build default rope for standard path
        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()

        # Native EliGen path for Qwen Image (entity prompt/mask conditioning)
        if eligen_entity_prompt_emb is not None and eligen_entity_prompt_emb_mask is not None and eligen_entity_masks is not None:
            encoder_hidden_states, image_rotary_emb, encoder_hidden_states_mask = self.process_entity_masks(
                latents=x,
                prompt_emb=encoder_hidden_states,
                prompt_emb_mask=attention_mask,
                entity_prompt_emb=eligen_entity_prompt_emb,
                entity_prompt_emb_mask=eligen_entity_prompt_emb_mask,
                entity_masks=eligen_entity_masks,
                img_ids=img_ids,
                base_image_seq_len=hidden_states.shape[1],
                transformer_options=transformer_options,
            )

        del ids, txt_ids, img_ids

        temb = self.time_text_embed(timestep, hidden_states, additional_t_cond)

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], timestep_zero_index=timestep_zero_index, transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    timestep_zero_index=timestep_zero_index,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        if timestep_zero_index is not None:
            temb = temb.chunk(2, dim=0)[0]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-3], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]
