import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .ops.modules import MSDeformAttn
from mmcv.cnn import build_norm_layer,build_norm_layer, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN,MultiheadAttention
from mmengine.model import ModuleList
from mmengine.model import BaseModule
from .utils.general_utils import ConfigType, OptMultiConfig,OptConfigType

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "PReLU":
        return build_activation_layer(dict(type='PReLU'))
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




class DeformableTransformerEncoderLayer(BaseModule):
    """Implements encoder layer in DETR transformer.
    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """
    def __init__(self,
                 self_attn_cfg: OptConfigType = dict(
                     embed_dims=256,n_levels=4, num_heads=8, n_points=4, dropout=0.0),
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = self_attn_cfg.embed_dims
        self.self_attn_cfg = self_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
            
        self.dropout1 = nn.Dropout(self_attn_cfg.dropout)
        
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MSDeformAttn(d_model=self.self_attn_cfg.embed_dims,
                                      n_levels=self.self_attn_cfg.n_levels,
                                      n_heads=self.self_attn_cfg.n_heads,
                                      n_points=self.self_attn_cfg.n_points)
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
    
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        complete_token = self.with_pos_embed(src, pos)
        bs, len_token, c = complete_token.shape
        sparse_token = torch.masked_select(complete_token, ~padding_mask.unsqueeze(-1))
        sparse_token = sparse_token.view(bs, -1, c)
        
        bs, len_token, nlevel, num_pts = reference_points.shape
        sparse_ref = torch.masked_select(reference_points, ~padding_mask.unsqueeze(-1).unsqueeze(-1))
        sparse_ref = sparse_ref.view(bs, -1, nlevel, num_pts)
        
        src2 = self.self_attn(sparse_token, sparse_ref, src, spatial_shapes, level_start_index, None)
        sparse_token = sparse_token + self.dropout1(src2)
        sparse_token = self.norms[0](sparse_token)
        
        sparse_token = self.ffn(sparse_token)
        sparse_token = self.norms[1](sparse_token)
    
        expanded_padding_mask = padding_mask.unsqueeze(-1).repeat(1,1,c)
        extended_sparse_token = torch.zeros_like(complete_token)
        extended_sparse_token[~expanded_padding_mask] = sparse_token.flatten()
        origin = complete_token.masked_fill(~expanded_padding_mask, 0)
        update = origin + extended_sparse_token
        return update
    


class DeformableTransformerEncoder(BaseModule):
    def __init__(self, num_layers,layer_cfg: ConfigType,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self._init_layers()
        
    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        Intermedia:
            - reference_points: [bs, sum(hi*wi), num_level, 2]
        """

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,linear_sample=False):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points,linear_sample)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, num_layers,num_points,query_dim, return_intermediate=False, use_dab=True,
                 no_sine_embed=False,layer_cfg=None, temperature=10000):
        super().__init__()
        self.layer_cfg = layer_cfg
        decoder_layer = DeformableTransformerDecoderLayer(**layer_cfg)
        self.layers = _get_clones(decoder_layer, num_layers)
        self.linear_sample = self.layer_cfg.linear_sample
        d_model = layer_cfg.d_model
        self.num_layers = num_layers
        self.query_dim = query_dim
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        self.num_points = num_points
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, d_model, d_model, 3)
            else:
                self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.offset_map = MLP(self.num_points,(self.num_points+1)//2,1,2)
        self.post_norm = build_norm_layer(dict(type='LN'),d_model)[1]
        self.temperature = temperature
    def forward(self, tgt, reference_points, src, src_spatial_shapes,       
                src_level_start_index, src_valid_ratios,
                offset_points, reg_branches,  
            offset_branches, query_pos=None, src_padding_mask=None):
        '''
        reference_points   (bs, n,3)
        offset_points       (bs, n,72)
        
        tgt nn.embedding   (bs,n,c)
        query_pos         () MLP(sine_emd(reference_points))
        
        Input:
            - src: [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - padding_mask: [bs, sum(hi*wi)]
        '''
        query_dim = self.query_dim
        assert  query_dim==3, f"query_dim==3 is not true"
        output = tgt
        if self.use_dab:
            assert query_pos is None
            
        #transform yxt to xyt,  and transform coordinates
        reference_points = torch.cat((reference_points[..., 1,None], 1-reference_points[..., 0,None], reference_points[..., 2,None]), dim=-1)
        xyt = reference_points
        intermediate = []
        intermediate_offest_points = []
        intermediate_yxt = []
        offset_map = self.offset_map(offset_points)

        reference_points = torch.cat([xyt,offset_map],dim=-1)  #bs  nq, 4(xytl)
        
        for lid, layer in enumerate(self.layers):
            #coordinates system left_top (0,0)
            if reference_points.shape[-1] == 4:
                ## bs, nq, 4(level), 4               [bs, nq,1, 4][bs, 1, 4,1]
                reference_points_input = reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None] # bs, nq, 4(level), 4  
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.use_dab:
                if self.no_sine_embed:
                    raw_query_pos = self.ref_point_head(reference_points_input)
                else:
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :],  
                                                                  temperature=self.temperature) # bs, nq, 256*2 
                    raw_query_pos = self.ref_point_head(query_sine_embed) # bs, nq, 256
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            

            if self.linear_sample:
                output = layer(output, query_pos, reference_points_input[...,:3], src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            else:
                output = layer(output, query_pos, reference_points_input[...,:2], src, src_spatial_shapes, src_level_start_index, src_padding_mask)
                

            new_reference_points = torch.zeros_like(reference_points)
            
            tmp_yxt = reg_branches(output) 
            tmp_xyt = torch.cat((tmp_yxt[..., 1,None], tmp_yxt[..., 0,None], tmp_yxt[..., 2,None]), dim=-1)
            new_xyt = tmp_xyt + inverse_sigmoid(reference_points[...,:query_dim]) #delta + xyt
            new_xyt = new_xyt.sigmoid()
            yxt = torch.cat((1-new_xyt[..., 1,None], new_xyt[..., 0,None], new_xyt[..., 2,None]), dim=-1)
            if self.return_intermediate:
                intermediate_yxt.append(yxt)
            new_reference_points[...,:query_dim] = new_xyt.detach()


            tmp_offset_points = offset_branches(output)
            tmp_offset_points += offset_points
            if self.return_intermediate:
                intermediate_offest_points.append(tmp_offset_points)
            offset_points = tmp_offset_points.detach()
            new_reference_points[...,query_dim:] = self.offset_map(offset_points) # map offset to an query
            
            reference_points = new_reference_points.detach()
            
            if self.return_intermediate:
                intermediate.append(self.post_norm(output))
        output = self.post_norm(output)

        
        if self.return_intermediate:
            return [
                torch.stack(intermediate),
                torch.stack(intermediate_yxt),
                torch.stack(intermediate_offest_points)
            ]
        else:
            return [
                output.unsqueeze(0),
                yxt.unsqueeze(0),
                tmp_offset_points.unsqueeze(0)
            ]






class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def gen_sineembed_for_position(pos_tensor, temperature=10000):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos