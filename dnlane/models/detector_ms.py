# This file is mainly modified from DAB-DETR in mmdetection
from typing import Tuple,Dict,List
import cv2
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import normal_

import mmcv
from mmdet.models.detectors import BaseDetector
from mmdet.models import build_backbone,build_head,build_neck
from mmdet.models.builder import MODELS
from mmengine.model import uniform_init

from .transformer import SinePositionalEncoding
from .deform_transformer import DeformableTransformerEncoder, DeformableTransformerDecoder
from .utils.general_utils import COLORS
import time

def lane_token_selector(seg_feature, seg_decoder, sparse_alpha, img_feats):
    '''
    输入seg_feature: (bs,5,320,800)  0通道为背景类
    返回mask_flatten: (bs,sum(hi*wi))  稀疏度严格为sparse_alpha的多尺度mask, 从大到小
    '''
    
    seg_feature = seg_feature.detach()
    seg = seg_decoder(seg_feature)  #(bs,5,320,800)  0通道为背景类
    bs,_,seg_h,seg_w = seg.shape
    seg = torch.softmax(seg, 1).flatten(2) #(bs,5,320*800)
    mask_flatten = []
   
    
    merged_mask, indices = torch.max(seg[:,1:], dim=1, keepdim=True)
    nlevel = len(img_feats)
    for l, l_src in enumerate(img_feats):
        bs, c, h,w = l_src.shape
        masks = F.interpolate(
        merged_mask.view(bs,-1,seg_h,seg_w), size=(h, w), mode='bilinear').flatten(1).unsqueeze(1)
        #FALSE为前景，
        threshold = torch.topk(masks, int(masks.size(2)  * sparse_alpha),\
                            largest=True).values[:, :, -1]
        masks = (masks > threshold.unsqueeze(2)).to(torch.bool)
                
        temp_mask = masks.flatten(1)
        # 检查是否有全零的样本, 即全是背景类
        sum_temp_mask = temp_mask.sum(1)
        all_zeros_mask = sum_temp_mask == 0
        # 对于全零的样本，随机生成一个稀疏度sparse_alpha 的前景 mask， lane True
        for i in range(bs):
            if all_zeros_mask[i]:
                random_mask_indices = torch.randperm(h * w )[:int(sparse_alpha * h * w)]
                temp_mask[i, random_mask_indices] = True
        
        
        # 计算每个通道中 1 的数量
        num_ones_per_channel = temp_mask.sum(dim=1)
        # 找到最小的数量，作为所有通道的标准
        min_ones_per_channel = num_ones_per_channel.min()
        # 将每个通道中超出最小数量的部分设置为 0
        for i in range(temp_mask.size(0)):
            num_ones = num_ones_per_channel[i]
            if num_ones > min_ones_per_channel:
                excess_ones = num_ones - min_ones_per_channel
                excess_indices = (temp_mask[i] == 1).nonzero().flatten()[:excess_ones]
                temp_mask[i][excess_indices] = False
        mask_flatten.append(~temp_mask)
    mask_flatten = torch.cat(mask_flatten, 1)
    return mask_flatten

@MODELS.register_module()
class MSLATR(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 encoder = None,
                 decoder = None,
                 num_queries = None,
                 offset_dim = None,
                 positional_encoding = None,
                 pretrain = None,
                 with_random_refpoints = False,
                 num_patterns = 0,
                 train_cfg = None,
                 test_cfg = None,
                 left_prio = 1,   
                 sample_y = range(589, 270, -8), 
                 num_feat_layers = 3,
                 sparse_alpha = 1,               
                 **kwargs
                 ):
        super().__init__()

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.bbox_head = build_head(head)

        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.with_random_refpoints = with_random_refpoints
        self.left_prio = left_prio
        self.sample_y = sample_y
        self.sparse_alpha = sparse_alpha
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableTransformerEncoder(**self.encoder)
        self.decoder = DeformableTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_dim = self.decoder.query_dim
        self.n_offsets = self.bbox_head.n_offsets
        self.query_embedding = nn.Embedding(self.num_queries, self.query_dim)
        #self.level_embed = nn.Embedding(num_feat_layers, self.embed_dims)
        self.num_feat_layers = num_feat_layers
        self.level_embed = nn.Parameter(torch.Tensor(num_feat_layers, self.embed_dims)) #no gradient
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, self.embed_dims)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
            
        self.padding_mask = None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()
    
    def init_weights(self) -> None:
        """Initialize weights for query according to the left/right/bottom position."""
        super(BaseDetector, self).init_weights()
        normal_(self.level_embed)
        left_priors_nums = self.left_prio
        bottom_priors_nums = self.num_queries - 2*left_priors_nums
        assert bottom_priors_nums >0,"bottom_priors_nums should be greater than 0"
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)
        for i in range(left_priors_nums):
            nn.init.constant_(self.query_embedding.weight[i, 0],
                              (i // 2) * strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 1], 0.)
            nn.init.constant_(self.query_embedding.weight[i, 2],
                              0.16 if i % 2 == 0 else 0.32)
        
        #bot query
        for i in range(left_priors_nums,
                       left_priors_nums + bottom_priors_nums):
            nn.init.constant_(self.query_embedding.weight[i, 0], 0.)
            nn.init.constant_(self.query_embedding.weight[i, 1],
                              ((i - left_priors_nums) // 4 + 1) *
                              bottom_strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 2],
                              0.2 * (i % 4 + 1))
        #right query初始化
        for i in range(left_priors_nums + bottom_priors_nums, self.num_queries):
            nn.init.constant_(
                self.query_embedding.weight[i, 0],
                ((i - left_priors_nums - bottom_priors_nums) // 2) *   #y
                strip_size)
            nn.init.constant_(self.query_embedding.weight[i, 1], 1.)   #x
            nn.init.constant_(self.query_embedding.weight[i, 2],       #t
                              0.68 if i % 2 == 0 else 0.84)
        #self.query_embedding.weight.requires_grad = False
        
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.
        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).
        Returns:
            List[Tensor]: 
            output: [(B,256,h/8,w/8),(B,256,h/16,w/16),(B,256,h/32,w/32)]   
            x: [(B,128,h/8,w/8),(B,256,h/16,w/16),(B,512,h/32,w/32)]
        """
        x = self.backbone(batch_inputs)
        output = self.neck(x[-self.num_feat_layers:])
        
        return output,x   

    def pre_transformer(
            self,
            img_feats: List[Tensor],
            img_metas) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.
        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        Args:
            img_feats (List[Tensor]): List of features output from the neck,
                has shape (nlevels,). [(bs,c, h1, w1), (bs, c, h2, w2), (bs, c, h3, w3) ] 从大到小
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.
            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

       
        
        nlevel = len(img_feats)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        spatial_shapes = []
        
        nest_srcs = img_feats

        feat = nest_srcs[-1][0]  #c h/32 w32
        batch_size = len(nest_srcs[-1])
        batch_input_shape = img_metas[0]['img_metas']['image_shape']
        img_shape_list = [sample['img_metas']['image_shape'] for sample in img_metas]
        #print(f"img_shape_list{img_shape_list} batch{batch_size}")
        input_img_h, input_img_w = batch_input_shape
        minimal_masks = feat.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            minimal_masks[img_id, :img_h, :img_w] = 0
        # NOTE following the official DETR repo, non-zero values represent
        # ignored positions, while zero values mean valid positions.
        minimal_masks = F.interpolate(
            minimal_masks.unsqueeze(1), size=feat.shape[-2:]).squeeze(1)
        
        
        for l, l_src in enumerate(nest_srcs):

            batch_size, c, h, w = l_src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = l_src.flatten(2).transpose(1, 2)
            src_flatten.append(src)  # [bs, c, h, w] -> [bs, h*w, c]
            masks = F.interpolate(
            minimal_masks.unsqueeze(1), size=(h, w)).to(torch.bool).squeeze(1)
            temp_mask = masks.flatten(1)
            mask_flatten.append(temp_mask)
            
            pos_embed = self.positional_encoding(masks)
            #check self.level_embed[l].view(1, 1, -1) is constant
            pos_embed = pos_embed.flatten(2).transpose(1, 2) +  self.level_embed[l].view(1, 1, -1)  # [batch_size, h* w, embed_dim]
            lvl_pos_embed_flatten.append(pos_embed)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.ones((batch_size,nlevel,2),device=src_flatten.device)
        encoder_inputs_dict = dict(src=src_flatten, spatial_shapes=spatial_shapes,
                                   level_start_index=level_start_index,
                                   padding_mask=mask_flatten,
                                   pos=lvl_pos_embed_flatten,valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(memory_mask=mask_flatten,
                                   memory_pos=lvl_pos_embed_flatten,
                                   spatial_shapes=spatial_shapes,
                                   level_start_index=level_start_index,
                                   valid_ratios=valid_ratios)
        #decoder_inputs_dict = dict(memory_mask=minimal_masks.flatten(1).to(torch.bool), memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict
    

    
    def forward_encoder(self, src, spatial_shapes, level_start_index,
                        valid_ratios, pos=None, padding_mask=None):
        """Forward with Transformer encoder.
        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(src, spatial_shapes, level_start_index,
                        valid_ratios, pos, padding_mask) # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict
    
    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.
        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.
            - decoder_inputs_dict (dict): The keyword args dictionary of
                `self.forward_decoder()`, which includes 'query', 'query_pos',
                'memory' and 'reg_branches'.
            - head_inputs_dict (dict): The keyword args dictionary of the
                bbox_head functions, which is usually empty, or includes
                `enc_outputs_class` and `enc_outputs_class` when the detector
                support 'two stage' or 'query selection' strategies.
        """
        batch_size = memory.size(0)
        query_pos = self.query_embedding.weight
        #(bs, nq, 3)repeat query for batch input
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1) 
        if self.num_patterns == 0:
            query = query_pos.new_zeros(batch_size, self.num_queries,
                                        self.embed_dims)
        else:
            query = self.patterns.weight[:, None, None, :]\
                .repeat(1, self.num_queries, batch_size, 1)\
                .view(-1, batch_size, self.embed_dims)\
                .permute(1, 0, 2)
            query_pos = query_pos.repeat(1, self.num_patterns, 1)
        offset_points = query.new_zeros((batch_size, self.num_queries, self.n_offsets),device=query.device)
        
        decoder_inputs_dict = dict(
            query_pos=query_pos, query=query, memory=memory,offset_points=offset_points)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def forward_decoder(self, query: Tensor, query_pos: Tensor, memory: Tensor,
                        memory_mask: Tensor, memory_pos: Tensor, offset_points: Tensor,
                        spatial_shapes: Tensor,level_start_index: Tensor,
                        valid_ratios: Tensor):
        """Forward with Transformer decoder.
        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).
        Returns:
            head_inputs_dict: The dictionary of decoder outputs, which includes the
            `hidden_states` and `references  x y theta` and offset_points dim=72 of the decoder output.


        """
        offset_branch = nn.Sequential(self.bbox_head.feat_layer,self.bbox_head.reg_layers)
        hidden_states, references, offset_points = self.decoder(
            tgt=query,
            reference_points=query_pos, 
            src=memory,
            src_spatial_shapes=spatial_shapes,
            src_level_start_index=level_start_index,
            src_valid_ratios=valid_ratios,
            offset_points = offset_points,
            reg_branches=self.bbox_head.fc_reg,  # iterative refinement for anchor boxes
            offset_branches=offset_branch,      # iterative refinement for offset
            query_pos=None,
            src_padding_mask=memory_mask
     
        )
        head_inputs_dict = dict(
            hidden_states=hidden_states, references=references, offset_points = offset_points)
        return head_inputs_dict 
    
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            img_metas,
                            batch_feature : Tuple[Tensor] = None,) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:
        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.
        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        batch_feature = list(batch_feature)
        seg_feature = torch.cat([
                F.interpolate(feature,
                              size=[
                                  batch_feature[0].shape[2],
                                  batch_feature[0].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_feature
            ],dim=1)  
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, img_metas)
        if self.sparse_alpha != 1:
            padding_mask = lane_token_selector(seg_feature, self.bbox_head.seg_decoder, self.sparse_alpha, img_feats)
            encoder_inputs_dict.update({"padding_mask":padding_mask})
        self.padding_mask = encoder_inputs_dict["padding_mask"]
        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict)
        decoder_inputs_dict.update(tmp_dec_in)
        
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update({"seg_feature":seg_feature})
        head_inputs_dict.update({"img_metas":img_metas})
        return head_inputs_dict

    def forward_train(self, img, img_metas = None, **kwargs):
        targets = kwargs['lane_line']
        seg_targets = kwargs['seg']
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)
        head_out = self.bbox_head(**head_inputs_dict)
        head_out.update({"targets":targets,"seg_targets":seg_targets})
        loss = self.bbox_head.loss(**head_out)
        return loss

    def train_step(self, data, optimizer):
        losses = self.forward_train(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
        return outputs

    def predict(self,img, img_metas = None,**kwargs):
        img = img.cuda()
        img_metas = img_metas.data[0]
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)   
        out_head = self.bbox_head(**head_inputs_dict)
        output = self.bbox_head.get_lanes(out_head)
        return output

    def forward_dummy(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        shape = img.shape[2:]
        img_metas = [{"img_metas":{"image_shape":shape}}]
        img_feats,batch_feature = self.extract_feat(img)
        head_inputs_dict = self.forward_transformer(img_feats,img_metas,batch_feature)  
        out_head = self.bbox_head(**head_inputs_dict)
#        output = self.bbox_head.get_lanes(out_head)
        return out_head

    def show_result(self,img, lanes, show=False, out_file=None, width=4):
        """
        Draw detection lane over image
        """
        img = mmcv.imread(img)
        lanes = lanes[0]
        lanes = [lane.to_array(self.sample_y,self.bbox_head.ori_img_w,self.bbox_head.ori_img_h) for lane in lanes]
        lanes_xys = []
        for _, lane in enumerate(lanes):
            xys = []
            for x, y in lane:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                xys.append((x, y))
            lanes_xys.append(xys)
        lanes_xys = [xys for xys in lanes_xys if xys!=[]]

        for idx, xys in enumerate(lanes_xys):
            for i in range(1, len(xys)):
                # cv2.circle(img, xys[i], radius=2, color=COLORS[idx], thickness=width)
                cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)

        if show:
            cv2.imshow('view', img)
            cv2.waitKey(0)

        if out_file:
            cv2.imwrite(out_file, img)
    
    def simple_test(self, img, img_metas, **kwargs):
        return super().simple_test(img, img_metas, **kwargs)
    
    def aug_test(self, imgs, img_metas, **kwargs):
        return super().aug_test(imgs, img_metas, **kwargs)
