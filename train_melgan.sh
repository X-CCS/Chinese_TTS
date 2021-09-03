CUDA_VISIBLE_DEVICES=1 python examples/melgan/train_melgan.py \
    --train-dir ./dump_wanmei/train/ \
    --dev-dir ./dump_wanmei/valid/ \
    --outdir ./examples/melgan/exp/train.melgan.v1/ \
    --config ./examples/melgan/conf/melgan.v1.yaml \
    --use-norm 1 \
    --generator_mixed_precision 0 \
    --resume ""
