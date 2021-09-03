CUDA_VISIBLE_DEVICES=3 python examples/parallel_wavegan/train_parallel_wavegan.py \
    --train-dir ./dump_wanmei/train/ \
    --dev-dir ./dump_wanmei/valid/ \
    --outdir ./examples/parallel_wavegan/exp/train.parallel_wavegan.v1/ \
    --config ./examples/parallel_wavegan/conf/parallel_wavegan.v1.yaml \
    --use-norm 1 \
    --generator_mixed_precision 1 \
    --resume ""
