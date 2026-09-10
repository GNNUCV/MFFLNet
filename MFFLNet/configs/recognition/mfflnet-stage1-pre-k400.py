_base_ = ['../_base_/default_runtime.py']
# model settings
num_frames = 16
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='MFFLNet',
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        t_size=num_frames,
        dw_reduction=1.5,
        backbone_drop_path_rate=0.,
        temporal_downsample=False,
        no_mffl=True,
        n_dim=768,
        clip_pretrained=True,
        pretrained='ViT-B/16',
        freeze_backbone=True, # stage1开启主干冻结
    ),
    cls_head=dict(
        type='UniFormerHead',
        dropout_ratio=0.3,
        num_classes=3,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[114.75, 114.75, 114.75],
        std=[57.375, 57.375, 57.375],
        format_shape='NCTHW'))
# dataset settings
dataset_type = 'VideoDataset'
data_root = '/nfs/ysz/dataset'
data_root_val = '/nfs/ysz/dataset'
ann_file_train = '/nfs/ysz/dataset/FSVR/train_labels.csv'
ann_file_val = '/nfs/ysz/dataset/FSVR/val_labels.csv'
ann_file_test = '/nfs/ysz/dataset/FSVR/test_labels.csv'
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=num_frames, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSample', clip_len=num_frames, num_clips=2,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

randomness = dict(deterministic=True, seed=220)
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ========= Stage1优化器&调度器 =========
base_lr = 2e-5
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            # 主干全部置0，双重保险（虽然代码已经requires_grad=False）
            'transformer.resblocks': dict(lr_mult=0.0, decay_mult=0.0),
            'positional_embedding': dict(lr_mult=0.0),
            'class_embedding': dict(lr_mult=0.0),
            'conv1': dict(lr_mult=0.0),
            'mffl': dict(lr_mult=1.0), # mffl完整学习率
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2), # 调小梯度裁剪，原20过大
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6 / base_lr,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=7,
        eta_min=base_lr * 0.01,
        by_epoch=True,
        begin=3,
        end=10,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=100),
    logger=dict(interval=100),
)

auto_scale_lr = dict(enable=False, base_batch_size=256) # 关闭自动lr缩放，小数据集容易乱缩放
