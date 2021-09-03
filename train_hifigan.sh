CUDA_VISIBLE_DEVICES=2 python examples/hifigan/train_hifigan.py \
    --train-dir ./dump_wanmei/train/ \
    --dev-dir ./dump_wanmei/valid/ \
    --outdir ./examples/hifigan/exp/train.hifigan.v1/ \
    --config ./examples/hifigan/conf/hifigan.v1.yaml \
    --use-norm 1 \
    --generator_mixed_precision 1 \
    --resume ""
