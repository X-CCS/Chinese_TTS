CUDA_VISIBLE_DEVICES=3 python examples/multiband_melgan/train_multiband_melgan.py \
    --train-dir ./dump_wanmei/train/ \
    --dev-dir ./dump_wanmei/valid/ \
    --outdir ./examples/multiband_melgan/exp/train.multiband_melgan.v2/ \
    --config ./examples/multiband_melgan/conf/multiband_melgan.baker.v1.yaml \
    --use-norm 1 \
    --generator_mixed_precision 1 \
    --resume "./examples/multiband_melgan/exp/train.multiband_melgan.v2/checkpoints/ckpt-2680000"
