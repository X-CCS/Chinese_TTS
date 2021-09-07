import os
import sys

import tensorflow as tf

import yaml
import numpy as np

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

fastspeech2_config_path = "./origin/fastspeech2wanmei.yaml"
fastspeech2_config = AutoConfig.from_pretrained(fastspeech2_config_path)
fastspeech2 = TFAutoModel.from_pretrained("./origin/fastspeech2.h5", fastspeech2_config, name="fastspeech2")




