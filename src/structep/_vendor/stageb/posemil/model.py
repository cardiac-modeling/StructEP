"""Checkpoint-compatible StructEP model core.

The registered architecture combines a Morgan-fingerprint ligand branch with a
Transformer protein-ligand pose encoder. Pose evidence is pooled within each
receptor state, then across states, before gated fusion with the 2D embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .heads import HeadsConfig, PoseMILHeads
from .ligand2d import Ligand2DBranch, Ligand2DConfig
from .pose_mil import PoseAttentionMIL, PoseMILConfig, _segment_softmax
from .transformer_backbone import ProteinLigandTransformer, TransformerBackboneConfig


DEFAULT_CHANNELS = ("hERG", "NaV1.5", "CaV1.2")


@dataclass
class TriChannelPoseMILConfig:
    """Configuration fields used by the released StructEP checkpoints."""

    morgan_bits: int = 2048
    descriptor_dim: int = 0
    two_d_hidden_1: int = 1024
    two_d_hidden_2: int = 512
    d_model: int = 128
    dropout: float = 0.1
    backbone_type: str = "transformer"
    transformer_protein_layers: int = 3
    transformer_ligand_layers: int = 2
    transformer_fusion_layers: int = 2
    transformer_nhead: int = 4
    quality_dim: int = 5
    quality_hidden: int = 32
    pose_pooling: str = "attention"
    use_pose_quality: bool = True
    channels: Tuple[str, ...] = DEFAULT_CHANNELS
    channel_token_dim: int = 32
    use_channel_token: bool = False
    head_type: str = "standard"
    heads_hidden: int = 256
    ordinal_thresholds: Tuple[float, ...] = (5.0, 6.0, 7.0, 8.0)
    ordinal_value_clip: Tuple[float, float] = (3.0, 10.0)
    ordinal_residual_scale: float = 1.0
    log_var_clip: Tuple[float, float] = (-4.0, 2.0)
    per_channel_heads: bool = False
    fusion_gate_hidden: int = 128
    use_3d_branch: bool = True
    fusion_mode: str = "cf_gated"
    state_aware_3d: bool = True
    state_pooling: str = "attention"
    state_feature_dim: int = 0
    state_type_emb_dim: int = 0
    state_gate_hidden: int = 128
    state_aux_logit_bias: float = 0.0
    open_inact_delta_scale: float = 0.0
    assay_adapter_enabled: bool = False
    dual_gaussian_enabled: bool = False
    dual_gaussian_eta_max: float = 0.0
    use_ifp: bool = False
    ifp_input_dim: int = 0
    ifp_aux_enabled: bool = False
    ifp_aux_dim: int = 0
    use_maccs_residual_fusion: bool = False
    use_plec_head_residual: bool = False
    plec_sidecar_dim: int = 0


def _coerce_tuple(values: Sequence[str], name: str) -> Tuple[str, ...]:
    if values is None:
        raise ValueError(f"{name} must be a non-empty sequence")
    result = tuple(str(value) for value in values)
    if not result:
        raise ValueError(f"{name} must be a non-empty sequence")
    return result


class StateAwarePoseAggregator(nn.Module):
    """Pool pose embeddings within states and state embeddings within a bag."""

    OPEN_IDX = 1
    INACT_IDX = 2

    def __init__(
        self,
        *,
        d_model: int,
        state_feature_dim: int,
        state_type_emb_dim: int,
        quality_dim: int,
        hidden: int,
        dropout: float,
        aux_logit_bias: float,
        open_inact_delta_scale: float,
        herg_index: int,
        pooling_mode: str = "attention",
    ) -> None:
        super().__init__()
        self.state_feature_dim = int(state_feature_dim)
        self.quality_dim = int(quality_dim)
        self.state_type_emb = nn.Embedding(4, state_type_emb_dim)
        self.aux_logit_bias = float(aux_logit_bias)
        self.open_inact_delta_scale = float(open_inact_delta_scale)
        self.herg_index = int(herg_index)
        self.pooling_mode = str(pooling_mode or "attention").lower()
        if self.pooling_mode not in {"attention", "mean"}:
            raise ValueError(
                f"state pooling_mode must be 'attention' or 'mean', got {pooling_mode!r}"
            )

        gate_in = (
            d_model * 4
            + 1
            + self.quality_dim
            + self.state_feature_dim
            + state_type_emb_dim
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_in),
            nn.Linear(gate_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.delta = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        *,
        z_2d: torch.Tensor,
        z_state: torch.Tensor,
        pose_attn: torch.Tensor,
        pose_quality: Optional[torch.Tensor],
        state_index: torch.Tensor,
        state_to_bag: torch.Tensor,
        state_features: Optional[torch.Tensor],
        state_type_idx: Optional[torch.Tensor],
        state_role: Optional[torch.Tensor],
        channel_idx: Optional[torch.Tensor],
        num_bags: int,
    ) -> Dict[str, torch.Tensor]:
        state_count = z_state.size(0)
        if state_count == 0:
            zero_bag = z_2d.new_zeros(z_2d.shape)
            return {
                "z_3d": zero_bag,
                "state_attention": z_2d.new_zeros((0,)),
                "state_logits": z_2d.new_zeros((0,)),
                "state_pose_entropy": z_2d.new_zeros((0,)),
                "state_quality_mean": z_2d.new_zeros((0, self.quality_dim)),
                "state_delta": zero_bag,
                "state_aux_mask": torch.zeros(
                    (0,), device=z_2d.device, dtype=torch.bool
                ),
                "state_aux_nonherg_mask": torch.zeros(
                    (0,), device=z_2d.device, dtype=torch.bool
                ),
            }

        state_index = state_index.to(device=z_state.device, dtype=torch.long)
        state_to_bag = state_to_bag.to(device=z_state.device, dtype=torch.long)
        z2d_state = z_2d[state_to_bag]

        if state_features is None:
            state_features = z_state.new_zeros(
                (state_count, self.state_feature_dim)
            )
        state_features = state_features.to(
            device=z_state.device, dtype=z_state.dtype
        )
        if state_features.size(-1) < self.state_feature_dim:
            padding = z_state.new_zeros(
                (state_count, self.state_feature_dim - state_features.size(-1))
            )
            state_features = torch.cat([state_features, padding], dim=-1)
        elif state_features.size(-1) > self.state_feature_dim:
            state_features = state_features[:, : self.state_feature_dim]

        type_idx = (
            state_type_idx
            if state_type_idx is not None
            else state_to_bag.new_zeros((state_count,))
        )
        type_idx = type_idx.to(device=z_state.device, dtype=torch.long)
        type_idx = type_idx.clamp(
            min=0, max=self.state_type_emb.num_embeddings - 1
        )
        type_emb = self.state_type_emb(type_idx)

        safe_attn = pose_attn.clamp_min(1e-8)
        entropy_per_pose = -(safe_attn * safe_attn.log())
        pose_entropy = z_state.new_zeros((state_count,))
        pose_entropy.index_add_(
            0, state_index, entropy_per_pose.to(dtype=pose_entropy.dtype)
        )
        counts = z_state.new_zeros((state_count,))
        counts.index_add_(
            0, state_index, torch.ones_like(pose_attn, dtype=counts.dtype)
        )
        pose_entropy = (
            pose_entropy / counts.clamp_min(1.0).log().clamp_min(1.0)
        ).clamp(0.0, 1.0)

        if pose_quality is None:
            quality_mean = z_state.new_zeros(
                (state_count, self.quality_dim)
            )
        else:
            pose_quality = pose_quality.to(
                device=z_state.device, dtype=z_state.dtype
            )
            if pose_quality.size(-1) < self.quality_dim:
                padding = z_state.new_zeros(
                    (
                        pose_quality.size(0),
                        self.quality_dim - pose_quality.size(-1),
                    )
                )
                pose_quality = torch.cat([pose_quality, padding], dim=-1)
            elif pose_quality.size(-1) > self.quality_dim:
                pose_quality = pose_quality[:, : self.quality_dim]
            quality_mean = z_state.new_zeros(
                (state_count, self.quality_dim)
            )
            quality_mean.index_add_(
                0, state_index, pose_quality.to(dtype=quality_mean.dtype)
            )
            quality_mean = quality_mean / counts.clamp_min(1.0).unsqueeze(-1)

        gate_input = torch.cat(
            [
                z2d_state,
                z_state,
                (z_state - z2d_state).abs(),
                z_state * z2d_state,
                pose_entropy.unsqueeze(-1),
                quality_mean,
                state_features,
                type_emb,
            ],
            dim=-1,
        )
        logits = self.gate(gate_input).squeeze(-1)
        state_aux_mask = logits.new_zeros((state_count,), dtype=torch.bool)
        state_aux_nonherg_mask = logits.new_zeros(
            (state_count,), dtype=torch.bool
        )
        herg_bag_mask = torch.zeros(
            (num_bags,), device=logits.device, dtype=torch.bool
        )
        if channel_idx is not None and self.herg_index >= 0:
            channel_per_bag = channel_idx.to(
                device=logits.device, dtype=torch.long
            )
            herg_bag_mask = channel_per_bag == self.herg_index

        if state_role is not None:
            state_role = state_role.to(
                device=logits.device, dtype=logits.dtype
            )
            state_aux_mask = state_role < 0.5
            logits = logits + (
                state_aux_mask.to(dtype=logits.dtype) * self.aux_logit_bias
            )
            if channel_idx is not None:
                if self.herg_index >= 0:
                    channel_per_state = channel_idx.to(
                        device=state_to_bag.device, dtype=torch.long
                    )[state_to_bag]
                    state_aux_nonherg_mask = state_aux_mask & (
                        channel_per_state != self.herg_index
                    )
                else:
                    state_aux_nonherg_mask = state_aux_mask

        if self.pooling_mode == "mean":
            states_per_bag = z_state.new_zeros((num_bags,))
            states_per_bag.index_add_(
                0,
                state_to_bag,
                torch.ones_like(logits, dtype=z_state.dtype),
            )
            alpha = 1.0 / states_per_bag[state_to_bag].clamp_min(1.0)
        else:
            alpha = _segment_softmax(logits, state_to_bag, num_bags)

        z_3d = z_state.new_zeros((num_bags, z_state.size(-1)))
        z_3d.index_add_(
            0,
            state_to_bag,
            (alpha.unsqueeze(-1) * z_state).to(dtype=z_3d.dtype),
        )
        if self.pooling_mode == "mean":
            state_delta = torch.zeros_like(z_3d)
        else:
            state_delta = self._open_inact_delta(
                z_state,
                alpha,
                state_to_bag,
                type_idx,
                num_bags,
                herg_bag_mask,
            )
        return {
            "z_3d": z_3d + state_delta,
            "state_attention": alpha,
            "state_logits": logits,
            "state_pose_entropy": pose_entropy,
            "state_quality_mean": quality_mean,
            "state_delta": state_delta,
            "state_aux_mask": state_aux_mask,
            "state_aux_nonherg_mask": state_aux_nonherg_mask,
        }

    def _open_inact_delta(
        self,
        z_state: torch.Tensor,
        alpha: torch.Tensor,
        state_to_bag: torch.Tensor,
        state_type_idx: torch.Tensor,
        num_bags: int,
        herg_bag_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dimension = z_state.size(-1)
        open_sum = z_state.new_zeros((num_bags, dimension))
        inact_sum = z_state.new_zeros((num_bags, dimension))
        open_weight = z_state.new_zeros((num_bags,))
        inact_weight = z_state.new_zeros((num_bags,))
        open_mask = state_type_idx == self.OPEN_IDX
        inact_mask = state_type_idx == self.INACT_IDX

        if bool(open_mask.any()):
            indices = state_to_bag[open_mask]
            weights = alpha[open_mask]
            open_sum.index_add_(
                0,
                indices,
                (weights.unsqueeze(-1) * z_state[open_mask]).to(
                    dtype=open_sum.dtype
                ),
            )
            open_weight.index_add_(0, indices, weights.to(dtype=open_weight.dtype))
        if bool(inact_mask.any()):
            indices = state_to_bag[inact_mask]
            weights = alpha[inact_mask]
            inact_sum.index_add_(
                0,
                indices,
                (weights.unsqueeze(-1) * z_state[inact_mask]).to(
                    dtype=inact_sum.dtype
                ),
            )
            inact_weight.index_add_(
                0, indices, weights.to(dtype=inact_weight.dtype)
            )

        open_embedding = open_sum / open_weight.clamp_min(1e-8).unsqueeze(-1)
        inact_embedding = (
            inact_sum / inact_weight.clamp_min(1e-8).unsqueeze(-1)
        )
        both = (
            (open_weight > 0) & (inact_weight > 0)
        ).to(dtype=z_state.dtype).unsqueeze(-1)
        delta_input = torch.cat(
            [
                open_embedding,
                inact_embedding,
                inact_embedding - open_embedding,
            ],
            dim=-1,
        )
        delta = (
            both
            * self.open_inact_delta_scale
            * self.delta(delta_input)
        )
        if herg_bag_mask is not None:
            delta = delta * herg_bag_mask.to(
                device=delta.device, dtype=delta.dtype
            ).unsqueeze(-1)
        return delta


class TriChannelPoseMIL(nn.Module):
    """Inference model used by every registered StructEP ensemble member."""

    def __init__(self, cfg: TriChannelPoseMILConfig) -> None:
        super().__init__()
        cfg.channels = _coerce_tuple(cfg.channels, "channels")
        self.cfg = cfg
        self._validate_runtime_config()

        self.ligand_2d = Ligand2DBranch(
            Ligand2DConfig(
                morgan_bits=cfg.morgan_bits,
                descriptor_dim=cfg.descriptor_dim,
                hidden_dim_1=cfg.two_d_hidden_1,
                hidden_dim_2=cfg.two_d_hidden_2,
                out_dim=cfg.d_model,
                dropout=cfg.dropout,
            )
        )
        self.complex_encoder = ProteinLigandTransformer(
            TransformerBackboneConfig(
                d_model=cfg.d_model,
                protein_layers=cfg.transformer_protein_layers,
                ligand_layers=cfg.transformer_ligand_layers,
                fusion_layers=cfg.transformer_fusion_layers,
                nhead=cfg.transformer_nhead,
                dropout=cfg.dropout,
            )
        )
        self.pose_mil = PoseAttentionMIL(
            PoseMILConfig(
                d_model=cfg.d_model,
                quality_dim=cfg.quality_dim,
                quality_hidden=cfg.quality_hidden,
                pooling_mode=cfg.pose_pooling,
                use_ifp=False,
                ifp_input_dim=0,
            )
        )
        herg_index = next(
            (
                index
                for index, channel in enumerate(cfg.channels)
                if str(channel).lower() == "herg"
            ),
            -1,
        )
        self.state_aggregator = StateAwarePoseAggregator(
            d_model=cfg.d_model,
            state_feature_dim=cfg.state_feature_dim,
            state_type_emb_dim=cfg.state_type_emb_dim,
            quality_dim=cfg.quality_dim,
            hidden=cfg.state_gate_hidden,
            dropout=cfg.dropout,
            aux_logit_bias=cfg.state_aux_logit_bias,
            open_inact_delta_scale=cfg.open_inact_delta_scale,
            herg_index=herg_index,
            pooling_mode=cfg.state_pooling,
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.fusion_gate_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_gate_hidden, cfg.d_model),
            nn.Sigmoid(),
        )
        self.channel_to_idx = {
            channel: index for index, channel in enumerate(cfg.channels)
        }
        self.heads = PoseMILHeads(
            HeadsConfig(
                input_dim=cfg.d_model,
                hidden_dim=cfg.heads_hidden,
                dropout=cfg.dropout,
                head_type=cfg.head_type,
                ordinal_thresholds=cfg.ordinal_thresholds,
                ordinal_value_clip=cfg.ordinal_value_clip,
                ordinal_residual_scale=cfg.ordinal_residual_scale,
                log_var_clip=cfg.log_var_clip,
            )
        )
        self.instance_score = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def _validate_runtime_config(self) -> None:
        required = {
            "backbone_type": "transformer",
            "head_type": "standard",
            "fusion_mode": "cf_gated",
            "use_3d_branch": True,
            "state_aware_3d": True,
            "per_channel_heads": False,
            "use_channel_token": False,
            "assay_adapter_enabled": False,
            "dual_gaussian_enabled": False,
            "use_ifp": False,
            "ifp_aux_enabled": False,
            "use_maccs_residual_fusion": False,
            "use_plec_head_residual": False,
        }
        for name, expected in required.items():
            observed = getattr(self.cfg, name)
            if observed != expected:
                raise ValueError(
                    f"StructEP requires {name}={expected!r}; found {observed!r}"
                )
        if len(self.cfg.channels) != 1:
            raise ValueError("StructEP checkpoints are channel-specific")

    def encode_2d(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.ligand_2d(batch["x_2d"])

    def encode_3d(
        self,
        batch: Dict[str, torch.Tensor],
        z_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        inst_emb, encoder_aux = self.complex_encoder(batch)
        pose_quality = (
            batch.get("pose_quality") if self.cfg.use_pose_quality else None
        )
        state_emb, pose_attn, content, quality, ifp_bias = self.pose_mil(
            inst_emb,
            batch["state_index"],
            int(batch["num_states"]),
            pose_quality=pose_quality,
            ifp_features=None,
        )
        state_output = self.state_aggregator(
            z_2d=z_2d,
            z_state=state_emb,
            pose_attn=pose_attn,
            pose_quality=pose_quality,
            state_index=batch["state_index"],
            state_to_bag=batch["state_to_bag"],
            state_features=batch.get("state_features"),
            state_type_idx=batch.get("state_type_idx"),
            state_role=batch.get("state_role"),
            channel_idx=batch.get("channel_idx"),
            num_bags=int(batch["num_bags"]),
        )
        instance_score = self.instance_score(inst_emb).squeeze(-1)
        return {
            "inst_emb": inst_emb,
            "inst_score": instance_score,
            "bag_emb": state_output["z_3d"],
            "bag_attn": pose_attn,
            "content_score": content,
            "quality_score": quality,
            "ifp_attention_bias": ifp_bias,
            "ifp_fusion_delta_norm": inst_emb.new_zeros((inst_emb.size(0),)),
            "bag_cons_index": batch["state_index"],
            "bag_cons_num_segments": int(batch["num_states"]),
            "z_lig_pool": encoder_aux["z_lig_pool"],
            "z_prot_pool": encoder_aux["z_prot_pool"],
            "state_emb": state_emb,
            "state_to_bag": batch["state_to_bag"],
            "state_type_idx": batch.get("state_type_idx"),
            "state_role": batch.get("state_role"),
            **state_output,
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_2d = self.encode_2d(batch)
        aux_3d = self.encode_3d(batch, z_2d=z_2d)
        z_3d = aux_3d["bag_emb"]
        gate = self.fusion_gate(torch.cat([z_2d, z_3d], dim=-1))
        z_3d_residual = gate * z_3d
        fused = z_2d + z_3d_residual
        head_output = self.heads(fused)

        disagreement = torch.linalg.vector_norm(
            z_2d - z_3d, dim=-1
        ) / (z_2d.size(-1) ** 0.5)
        state_to_bag = aux_3d["state_to_bag"]
        state_alpha = aux_3d["state_attention"]
        pose_entropy = z_2d.new_zeros((z_2d.size(0),))
        pose_entropy.index_add_(
            0,
            state_to_bag,
            (
                state_alpha * aux_3d["state_pose_entropy"]
            ).to(dtype=pose_entropy.dtype),
        )
        safe_alpha = state_alpha.float().clamp_min(1e-8)
        state_entropy = safe_alpha.new_zeros((z_2d.size(0),))
        state_entropy.index_add_(
            0, state_to_bag, -(safe_alpha * safe_alpha.log())
        )
        state_counts = safe_alpha.new_zeros((z_2d.size(0),))
        state_counts.index_add_(
            0, state_to_bag, torch.ones_like(safe_alpha)
        )
        state_gate_entropy = (
            state_entropy
            / state_counts.clamp_min(1.0).log().clamp_min(1.0)
        ).clamp(0.0, 1.0).to(dtype=z_2d.dtype)

        raw_mu = head_output["mu_pic50"]
        output: Dict[str, torch.Tensor] = {
            "z_2d": z_2d,
            "z_3d": z_3d,
            "z_3d_residual": z_3d_residual,
            "z": fused,
            "h": fused,
            "head_h": fused,
            "fusion_gate": gate,
            "fusion_gate_mean": gate.mean(dim=-1),
            "pose_entropy": pose_entropy,
            "state_gate_entropy": state_gate_entropy,
            "disagreement_2d_3d": disagreement,
            "mu_raw_pic50": raw_mu,
            **head_output,
            "mu_ligand_pic50": raw_mu,
            "output_residual_raw_delta": raw_mu.new_zeros(raw_mu.shape),
            "output_delta_3d_pic50": raw_mu.new_zeros(raw_mu.shape),
            "output_residual_gate": raw_mu.new_ones(raw_mu.shape),
            "output_residual_scaled_delta": raw_mu.new_zeros(raw_mu.shape),
            "assay_adapter_delta": raw_mu.new_zeros(raw_mu.shape),
            "channel_reliability_logit": raw_mu.new_zeros(raw_mu.shape),
            "channel_reliability": raw_mu.new_ones(raw_mu.shape),
            "bag_attn": aux_3d["bag_attn"],
            "content_score": aux_3d["content_score"],
            "quality_score": aux_3d["quality_score"],
            "ifp_attention_bias": aux_3d["ifp_attention_bias"],
            "ifp_fusion_delta_norm": aux_3d["ifp_fusion_delta_norm"],
            "inst_emb": aux_3d["inst_emb"],
            "inst_score": aux_3d["inst_score"],
            "bag_cons_index": aux_3d["bag_cons_index"],
            "state_emb": aux_3d["state_emb"],
            "state_to_bag": aux_3d["state_to_bag"],
            "state_attention": aux_3d["state_attention"],
            "state_logits": aux_3d["state_logits"],
            "state_pose_entropy": aux_3d["state_pose_entropy"],
            "state_quality_mean": aux_3d["state_quality_mean"],
            "state_delta": aux_3d["state_delta"],
            "state_type_idx": aux_3d["state_type_idx"],
            "state_role": aux_3d["state_role"],
            "state_aux_mask": aux_3d["state_aux_mask"],
            "state_aux_nonherg_mask": aux_3d["state_aux_nonherg_mask"],
        }
        output["bag_cons_num_segments"] = aux_3d["bag_cons_num_segments"]
        return output

    def rank_logits(
        self, left_h: torch.Tensor, right_h: torch.Tensor
    ) -> torch.Tensor:
        return self.heads.rank_logits(left_h, right_h)
