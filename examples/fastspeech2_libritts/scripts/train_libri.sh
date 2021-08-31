CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_libritts/train_fastspeech2.py \
  --train-dir ./dump_libritts/train/ \
  --dev-dir ./dump_libritts/valid/ \
  --outdir ./examples/fastspeech2_libritts/outdir_libri/ \
  --config ./examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
  --use-norm 1 \
  --f0-stat ./dump_libritts/stats_f0.npy \
  --energy-stat ./dump_libritts/stats_energy.npy \
  --mixed_precision 1 \
  --dataset_config preprocess/libritts_preprocess.yaml \
  --dataset_mapping dump_libritts/libritts_mapper.json \
  --dataset_stats dump_libritts/stats.npy
