import os
import argparse
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

import yaml
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from scipy.io import wavfile

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig 
from tensorflow_tts.models import TFPQMF, TFMelGANGenerator

processor = AutoProcessor.from_pretrained("./config/wanmei.json")


def get_mel_from_text_ids(input_ids, text2mel_model, text2mel_name):
    mel = None

    if text2mel_name == "tacotron":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )

        mel = mel_outputs
    elif text2mel_name == "fastspeech2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
        )

        mel = mel_outputs
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

    return mel


def do_synthesis_origin(mel, text2mel_model, vocoder_model, text2mel_name, vocoder_name):
    audio = None

    # vocoder part
    remove_end = 1
    audio = vocoder_model.inference(mel)[0, :-remove_end, 0]
    
    #if text2mel_name == "tacotron":
    #    return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    #else:
    #    return mel_outputs.numpy(), audio.numpy()

    return audio.numpy()

def do_synthesis_by_multi_melgan(mel, text2mel_model, text2mel_name):
    # load multi band melgan checkpoint
    checkpoint_path = "./result/Multi_band_MelGAN/discriminator-80000.h5"
    config_path = "./examples/multiband_melgan/conf/multiband_melgan.v1.yaml"
    
    config = AutoConfig.from_pretrained(config_path)

    multi_melgan = TFAutoModel.from_pretrained(checkpoint_path, config, name="mb_melgan")
    
    audio = multi_melgan.inference(mel)[0, :, 0]

    return audio

#def do_synthesis_by_wavegan(mel):


def synthesize(input_text, str_acoustic_model, str_voice_decoder):
    # text -> text ids
    input_ids = processor.text_to_sequence(input_text)

    # select acoustic model
    acoustic_model = None
    if str_acoustic_model is "fastspeech2":
        config_path = "examples/fastspeech2_wanmei/conf/fastspeech2wanmei.yaml"
        config = AutoConfig.from_pretrained(config_path)
        fastspeech2 = TFAutoModel.from_pretrained("./result/fastspeech2_wanmei/model-300000.h5", config, name="fastspeech2")
        acoustic_model = fastspeech2

    # text -> mel
    mel = get_mel_from_text_ids(input_ids, acoustic_model, str_acoustic_model)

    print("text -> mel finish !!!!!")

    audio = None

    # select voice decoder
    if str_voice_decoder is "origin":
        config_path = "./result/origin/config.yml"
        config = AutoConfig.from_pretrained(config_path)
        origin_melgan = TFAutoModel.from_pretrained("./result/origin/model.h5", config, name="mb_melgan")
        
        # do_synthesis
        audio = do_synthesis_origin(mel, acoustic_model, origin_melgan, str_acoustic_model, str_voice_decoder)

        # save wave
        if audio is not None:
            wavfile.write(os.path.join("./", "{}.wav".format("aa")), 22050, audio)

    elif str_voice_decoder == "multi-band-melgan":
        config_path = "./result/Multi_band_MelGAN/multiband_melgan.baker.v1.yaml"
        config = AutoConfig.from_pretrained(config_path)

        gan = TFAutoModel.from_pretrained("./result/Multi_band_MelGAN/generator-2580000.h5", config)

        audio = do_synthesis_origin(mel, acoustic_model, gan, str_acoustic_model, str_voice_decoder)

        # save wave
        if audio is not None:
            wavfile.write(os.path.join("./", "{}.wav".format("cc")), 22050, audio)

    elif str_voice_decoder == "wave-gan":
        config_path = "./result/wavegan/parallel_wavegan.v1.yaml"
        config = AutoConfig.from_pretrained(config_path)

        gan = TFAutoModel.from_pretrained("./result/wavegan/generator-400000.h5", config)

        audio = do_synthesis_origin(mel, acoustic_model, gan, str_acoustic_model, str_voice_decoder)

        # save wave
        if audio is not None:
            wavfile.write(os.path.join("./", "{}.wav".format("dd")), 22050, audio)

    elif str_voice_decoder == "hifi-gan":
        config_path = "./result/hifigan/hifigan.v1.yaml"
        config = AutoConfig.from_pretrained(config_path)

        gan = TFAutoModel.from_pretrained("./result/hifigan/generator-1880000.h5", config)

        audio = do_synthesis_origin(mel, acoustic_model, gan, str_acoustic_model, str_voice_decoder)

        # save wave
        if audio is not None:
            wavfile.write(os.path.join("./", "{}.wav".format("ee")), 22050, audio)

    print("mel -> wav finish !!!!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument( 
        "--text", 
        type=str, 
        default=None, 
        help="raw text to synthesize, for single-sentence mode only", 
    )

    parser.add_argument( 
        "--speaker_id", 
        type=int, 
        default=0, 
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only", 
    )

    parser.add_argument(
        "--acoustic_model",
        type=str,
        choices=["fastspeech2", "tacotron"],
        default="fastspeech2",
        help="acoustic model : text -> mel",
    )

    parser.add_argument(
        "--voice_decoder",
        type=str,
        choices=["origin", "melgan", "multi-band-melgan", "hifi-gan", "wave-gan"],
        default="origin",
        help="voice decoder : mel -> wav",
    )

    args = parser.parse_args()

    if args.text is None:
        print("input text is null !")
        exit(0)

    synthesize(args.text, args.acoustic_model, args.voice_decoder)

