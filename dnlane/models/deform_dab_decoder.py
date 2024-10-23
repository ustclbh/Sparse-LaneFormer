from typing import Tuple,List
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from .ops.modules import MSDeformAttn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN,MultiheadAttention
from mmengine.model import ModuleList
from mmengine.model import BaseModule
from .transformer_utils import coordinate_to_encoding,DetrTransformerDecoder,DetrTransformerDecoderLayer
from .utils.general_utils import ConfigType, OptMultiConfig,OptConfigType

def inverse_sigmoid(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Inverse function of sigmoid.
    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.
    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.
        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_(MLP):
    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.
        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        return x


class DeformDABDecoderLayer(DetrTransformerDecoderLayer):
    """Implements decoder layer in DAB-DETR transformer."""

    def _init_layers(self):
        """Initialize self-attention, cross-attention, FFN, normalization and
        others."""
        self.self_attn = nn.MultiheadAttention(self.self_attn_cfg.embed_dims, 
                                               self.self_attn_cfg.num_heads, 
                                               dropout=self.self_attn_cfg.attn_drop)  #normal multi-head attn
        self.cross_attn = MSDeformAttn(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
        self.keep_query_pos = self.cross_attn.keep_query_pos

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                ref_sine_embed: Tensor = None,
                self_attn_masks: Tensor = None,
                cross_attn_masks: Tensor = None,
                key_padding_mask: Tensor = None,
                is_first: bool = False,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query with shape [bs, num_queries,
                dim].
            key (Tensor): The key tensor with shape [bs, num_keys,
                dim].
            query_pos (Tensor): The positional encoding for query in self
                attention, with the same shape as `x`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            ref_sine_embed (Tensor): The positional encoding for query in
                cross attention, with the same shape as `x`.
                Defaults to None.
            self_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_masks (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
            is_first (bool): A indicator to tell whether the current layer
                is the first layer of the decoder.
                Defaults to False.
        Returns:
            Tensor: forwarded results with shape
            [bs, num_queries, dim].
        """

        query = self.self_attn(
            query=query,
            key=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_masks,
            **kwargs)
        query = self.norms[0](query)
        query = self.cross_attn(
            query, 
            reference_points,
            input_flatten, 
            input_spatial_shapes, 
            input_level_start_index,
            input_padding_mask=None)
        
        query = self.cross_attn(
            query=query,
            key=key,
            query_pos=query_pos,
            key_pos=key_pos,
            ref_sine_embed=ref_sine_embed,
            attn_mask=cross_attn_masks,
            key_padding_mask=key_padding_mask,
            is_first=is_first,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query

class DABDetrTransformerDecoder(DetrTransformerDecoder):
    """Decoder of DAB-DETR.
    Args:
        query_dim (int): The last dimension of query pos,
            4 for anchor format, 2 for point format.
            Defaults to 4.
        query_scale_type (str): Type of transformation applied
            to content query. Defaults to `cond_elewise`.
        with_modulated_hw_attn (bool): Whether to inject h&w info
            during cross conditional attention. Defaults to True.
    """

    def __init__(self,
                 *args,
                 query_dim: int = 4,
                 query_scale_type: str = 'cond_elewise',
                 with_modulated_hw_attn: bool = True,
                 num_points = 72,
                 **kwargs):

        self.query_dim = query_dim
        self.query_scale_type = query_scale_type
        self.with_modulated_hw_attn = with_modulated_hw_attn
        self.num_points = num_points

        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize decoder layers and other layers."""
        assert self.query_dim in [2, 3, 4], \
            f'{"dab-detr only supports anchor prior or reference point prior"}'
        assert self.query_scale_type in [
            'cond_elewise', 'cond_scalar', 'fix_elewise'
        ]

        self.layers = ModuleList([
            DABDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        embed_dims = self.layers[0].embed_dims
        self.embed_dims = embed_dims

        self.post_norm = build_norm_layer(self.post_norm_cfg, embed_dims)[1]
        if self.query_scale_type == 'cond_elewise':
            self.query_scale = MLP(embed_dims, embed_dims, embed_dims, 2)
        elif self.query_scale_type == 'cond_scalar':
            self.query_scale = MLP(embed_dims, embed_dims, 1, 2)
        elif self.query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(self.num_layers, embed_dims)
        else:
            raise NotImplementedError('Unknown query_scale_type: {}'.format(
                self.query_scale_type))

        self.ref_point_head = MLP((self.query_dim + 1) * (embed_dims // 2), embed_dims,
                                  embed_dims, 2)

        if self.with_modulated_hw_attn and self.query_dim == 4:
            self.ref_anchor_head = MLP(embed_dims, embed_dims, 2, 2)

        self.keep_query_pos = self.layers[0].keep_query_pos
        if not self.keep_query_pos:
            for layer_id in range(self.num_layers - 1):
                self.layers[layer_id + 1].cross_attn.qpos_proj = None
        
        self.offset_map = MLP(self.num_points,(self.num_points+1)//2,1,2)

    def forward(self,
                query: Tensor,
                key: Tensor,
                query_pos: Tensor,
                key_pos: Tensor,
                reg_branches: nn.Module,
                offset_branches:nn.Module,
                offset_points:Tensor,
                key_padding_mask: Tensor = None,
                **kwargs) -> List[Tensor]:
        """Forward function of decoder.
        Args:
            query (Tensor): The input query with shape (bs, num_queries, dim).
            key (Tensor): The input key with shape (bs, num_keys, dim).
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            reg_branches (nn.Module): The regression branch for dynamically
                updating references in each layer.
            key_padding_mask (Tensor): ByteTensor with shape (bs, num_keys).
                Defaults to `None`.
        Returns:
            List[Tensor]: forwarded results with shape (num_decoder_layers,
            bs, num_queries, dim) if `return_intermediate` is True, otherwise
            with shape (1, bs, num_queries, dim). references with shape
            (num_decoder_layers, bs, num_queries, 2/4).
        """
        output = query
        unsigmoid_references = query_pos.clone()      #(Nx3)  x y theta
        reference_points = unsigmoid_references
        offset_map = self.offset_map(offset_points)   # offset_points= num of points = 72
        query_pos_wo = torch.cat([query_pos,offset_map],dim=-1) #(x y theta  LOE)
        
        intermediate_reference_points = [reference_points]
        intermediate_offest_points = [offset_points]
#        intermediate_reference_points = []
        intermediate = []
        for layer_id, layer in enumerate(self.layers):
            #--- to produce dynamic anchor pos query---
            obj_center = query_pos_wo   # (x y theta  LOE) without backpropogation  
            ref_sine_embed = coordinate_to_encoding(
                coord_tensor=obj_center, num_feats=self.embed_dims // 2)
            query_pos = self.ref_point_head(
                ref_sine_embed)  # [bs, nq, 2c] -> [bs, nq, c]
            # For the first decoder layer, do not apply transformation
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # apply transformation
            ref_sine_embed = ref_sine_embed[
                ..., :self.embed_dims] * pos_transformation

            output = layer(
                output,
                key,
                query_pos=query_pos,
                ref_sine_embed=ref_sine_embed,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                is_first=(layer_id == 0),
                **kwargs)
            

            tmp_reg_preds = reg_branches(output)
            tmp_reg_preds[..., :self.query_dim] += reference_points  #(bs, n, 3(xyt))
            new_reference_points = tmp_reg_preds[
                ..., :self.query_dim]
            if layer_id != self.num_layers - 1:
                intermediate_reference_points.append(new_reference_points)
            query_pos_wo[...,:self.query_dim] = new_reference_points.detach()

            # iter update offest
            tmp_offset_points = offset_branches(output)  #(bs, n, 72)
            tmp_offset_points += offset_points
            if layer_id != self.num_layers - 1:
                intermediate_offest_points.append(tmp_offset_points)
            offset_points = tmp_offset_points.detach()
            query_pos_wo[...,self.query_dim:] = self.offset_map(tmp_offset_points) # map offset to an query


            if self.return_intermediate:
                intermediate.append(self.post_norm(output))

        output = self.post_norm(output)

        if self.return_intermediate:
            return [
                torch.stack(intermediate),   # (Nq*num_layer, bs, c)
                torch.stack(intermediate_reference_points), # (Nq*num_layer, bs, 3)
                torch.stack(intermediate_offest_points) # (Nq*num_layer, bs, 72)
            ]
        else:
            return [
                output.unsqueeze(0),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_offest_points)
            ]
