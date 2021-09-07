import os
import sys

import tensorflow as tf

import yaml
import numpy as np

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor


def convert_to_tflite(model, name="model", quantize=True):
    # Concrete Function
    concrete_function = model.inference_tflite.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_function]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    # converter.target_spec.supported_types = [tf.float16]
    # it has bug if you use tf.float16, see https://github.com/TensorSpeech/TensorFlowTTS/issues/346#issuecomment-728656417
    # This colab doesn't care about the latency, so it compressed the model with quantization. 8 bit run on desktop will slow.

    if not quantize:
        converter.target_spec.supported_types = [tf.float32]

    tflite_model = converter.convert()

    saved_path = name + '_quan.tflite' if quantize else name + '.tflite'
    
    # Save the TF Lite model.
    with open(saved_path, 'wb') as f:
        f.write(tflite_model)

    print('Model: %s size is %f MBs.' % (name, len(tflite_model) / 1024 / 1024.0) )
    
    return saved_path


fastspeech2_config_path = "./android/origin/fastspeech2wanmei.yaml"
fastspeech2_config = AutoConfig.from_pretrained(fastspeech2_config_path)
fastspeech2_convert = TFAutoModel.from_pretrained(
                          "./android/origin/fastspeech2.h5", 
                          fastspeech2_config, 
                          name="fastspeech2",
                          enable_tflite_convertible=True,
                      )

mb_melgan_config_path = "./android/origin/mb-melgan.yml"
mb_melgan_config = AutoConfig.from_pretrained(mb_melgan_config_path)
mb_melgan_convert = TFAutoModel.from_pretrained(
                        "./android/origin/mb-melgan.h5", 
                        mb_melgan_config, 
                        name="mb_melgan"
                    )

fastspeech2_tflite_path = convert_to_tflite(fastspeech2_convert, "fastspeech2")
mb_melgan_tflite_path = convert_to_tflite(mb_melgan_convert, "mb_melgan", quantize=False) # it sounds bad when quantize

print("convert model finish !!!!")

print("fastspeech2 tflite path :", fastspeech2_tflite_path)
print("mb_melgan tflite path :", mb_melgan_tflite_path)

