_base_ = ['./mfflnet‑stage1-pre-k400.py']

# 加载stage1训练好的权重
load_from = 'work_dirs/mfflnet‑fsvr‑stage1/best_top1_acc.pth'

# 模型：关闭主干冻结
model = dict(
    backbone=dict(
        freeze_backbone=False
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=1, val_interval=1
)

# ========= Stage2超参 =========
base_lr = 4e-5

# 优化器，分层学习率
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.05
    ),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            'positional_embedding': dict(lr_mult=0.5),
            'class_embedding': dict(lr_mult=0.5),
            'transformer.resblocks': dict(lr_mult=0.5), # 原始主干lr减半
            'mffl': dict(lr_mult=1.0), # MFFL完整lr
        }
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=2, # 梯度累积，等效扩大batch
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=base_lr * 0.01,
        by_epoch=True,
        begin=0,
        end=30)
]

# 增加EMA钩子
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=100),
    logger=dict(interval=100)
)

auto_scale_lr = dict(enable=False, base_batch_size=256)
