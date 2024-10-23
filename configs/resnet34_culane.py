_base_ = [
    './_base_/culane.py', './_base_/default_runtime.py'
]

num_classes = 4
num_points = 72

ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270
d_model = 256
dropout=0.
temperature= 10000

work_dir = ''
data_root = ''

dataset_type = 'CULaneDataset'


file_client_args = dict(backend='disk')

ori_img_w = 1640
ori_img_h = 590
img_w = 800
img_h = 320
cut_height = 270

train_pipeline = [
    dict(
        type='GenerateLaneLine',
        keys=['img', 'lane_line', 'seg',],
        img_info = (img_w,img_h),
        num_points = 72,
        max_lanes = 4,
        meta_keys = ['img_metas'],
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
            dict(name='HorizontalFlip', parameters=dict(p=1.0), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10)),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10)),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5))),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5)))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2)),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w)),
                 p=1.0),
        ]
    ),
    dict(type='ToTensor_', keys=['img', 'lane_line', 'seg', 'img_metas']),
]

test_pipeline = [
    dict(type='GenerateLaneLine',
         keys=['img',],
         meta_keys = ['img_metas'],
         img_info = (img_w,img_h),
         num_points = 72,
         max_lanes = 4,
         transforms=[
             dict(name='Resize',
                  parameters=dict(size=dict(height=img_h, width=img_w)),
                  p=1.0),
         ],
         training=False),
    dict(type='ToTensor_', keys=['img','img_metas']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'train',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'test',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        split = 'test',
        cut_height = 270,
        img_fo = (ori_img_h,ori_img_w),
        resize_img_info = (img_h,img_w),
        pipeline=test_pipeline))

neck_in_dim = [128,256,512]
num_feat_layers = 3
ckpt_timm = 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth'
model = dict(
    type='MSLATR',
    num_queries=192,
    left_prio=24,
    with_random_refpoints=False,
    num_patterns=0,
    max_lanes = num_classes,
    num_feat_layers = num_feat_layers,
    sparse_alpha = 1,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=ckpt_timm)),
    neck=dict(
        type='ChannelMapper',
        in_channels=neck_in_dim[-num_feat_layers:],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=None
        ),
    
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=d_model, n_heads=8, n_levels=num_feat_layers, n_points=16, dropout=dropout),
            ffn_cfg=dict(
                embed_dims=d_model,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=dropout,
                act_cfg=dict(type='PReLU')))),
    decoder=dict(
        num_layers=6,
        query_dim =3,
        num_points = num_points,
        return_intermediate=True,
        use_dab=True,
        temperature=temperature,
        layer_cfg=dict(
            d_model=d_model, d_ffn=2048,
            dropout=dropout, activation="PReLU",
            n_levels=num_feat_layers, n_heads=8, n_points=16,linear_sample=True)
        ),
    positional_encoding=dict(num_feats=d_model//2, temperature=temperature, normalize=True),
    head=dict(
        type='DNHeadv2',
        num_classes=num_classes,
        num_points = num_points,
        img_info=(img_h,img_w),
        ori_img_info = (ori_img_h,ori_img_w),
        cut_height = cut_height,
        assigner = dict(type='One2ManyLaneAssigner',
                        distance_cost = dict(type="Distance_cost",weight=3.),
                        cls_cost = dict(type='FocalLossCost')),
        loss_cls=dict(
            type='FocalLoss_py',
            gamma=2.0,
            alpha=0.25,
            use_sigmoid=True,
            loss_weight=2.0),
        loss_xyt = dict(type='SmoothL1Loss',loss_weight = 0.3),
        loss_iou=dict(type='Line_iou', loss_weight=2.0),
        loss_seg = dict(type='CrossEntropyLoss',loss_weight=1.0,ignore_index=255),
        test_cfg = dict(conf_threshold=0.5)),
     train_cfg = None,
     test_cfg = None
    )

# optimizer
base_lr = 0.0004
interval = 1
eval_step = 2
optimizer = dict(
    type='AdamW',
    lr=base_lr, 
    weight_decay=0.00001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)})
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
max_epochs = 50
runner = dict(
    type='EpochBasedRunner', max_epochs=max_epochs)

# learning rate
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=3600,
    warmup_ratio=0.01,
    min_lr=1e-08
     )


checkpoint_config = dict(interval=interval)

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.004,
        priority=49)
]


log_config = dict(interval=200)
auto_scale_lr = dict(base_batch_size=16, enable=False)
evaluation = dict(
    output_basedir = work_dir,
    save_best='auto',
    interval=eval_step,
    metric='mAP')



