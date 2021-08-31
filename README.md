声学模型训练

1. 根据 TextGrid 提取 duration文件：

python examples/mfa_extraction/txt_grid_parser.py 
       --yaml_path examples/fastspeech2_wanmei/conf/fastspeech2wanmei.yaml 
       --dataset_path ./data/align 
       --text_grid_path ./data/TextGrid 
       --output_durations_path ./data/durations

2.  Preprocessing

python tts-preprocess.py 
        --rootdir ./data/align 
        --outdir ./dump_wanmei 
        --config preprocess/wanmei_preprocess.yaml 
        --dataset wanmei

3.  Normalization

python tts-normalize.py 
        --rootdir ./dump_wanmei 
        --outdir ./dump_wanmei 
        --config preprocess/wanmei_preprocess.yaml 
        --dataset wanmei

4.  根据 duration 文件 筛选出 所有配对文件， 包括 energy wav pitch 等， train目录和 valid目录都要保证一致


5.  Train

