_base_ = './psgformer_r50_psg.py'

model = dict(bbox_head=dict(transformer=dict(
    decoder1=dict(type='DetrTransformerDecoder',
                    return_intermediate=True,
                    num_layers=9,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                        #   dropout_layer=None,
                            batch_first=False),
                        ffn_cfgs=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ffn_drop=0.0,
                        #   dropout_layer=None,
                            add_identity=True),
                        feedforward_channels=2048,
                        operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                        'ffn', 'norm'))),
),),)

evaluation = dict(interval=1,
                  metric='sgdet',
                  relation_mode=True,
                  classwise=True)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=[50, 100])
runner = dict(type='EpochBasedRunner', max_epochs=150)

project_name = 'psg_maskformer'
expt_name = 'psg_maskformer_r50_psg_entropy_rel_sop_gt'
work_dir = f'./work_dirs/{expt_name}'
checkpoint_config = dict(interval=10, max_keep_ckpts=15)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
                # config=work_dir + "/cfg.yaml"
            ),
        )
    ],
)

# load_from = './work_dirs/checkpoints/detr4psgformer_r50.pth'
# load_from = './work_dirs/checkpoints/psg_mask2former_trial.pth'
load_from = './work_dirs/checkpoints/psg_mask2former_dict.pth'