# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Perform preprocessing and raw feature extraction for LibriTTS dataset."""

import os
import re

import numpy as np
import soundfile as sf
from dataclasses import dataclass
from pypinyin import pinyin, Style

from tensorflow_tts.processor.base_processor import BaseProcessor
from tensorflow_tts.utils.utils import PROCESSOR_FILE_NAME

from tensorflow_tts.text.symbols import symbols
from tensorflow_tts.text import wanmei_text_to_sequence

WANMEI_SYMBOLS = symbols

zhongwen_pattern = re.compile("[\u4e00-\u9fa5]") 
 
def is_zh(word): 
    global zh_pattern 
    match = zhongwen_pattern.search(word)
    return match is not None

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

@dataclass
class WanmeiProcessor(BaseProcessor):

    mode: str = "train"
    train_f_name: str = "train.txt"
    positions = {
        "file": 0,
        "text": 1,
        "speaker_name": 2,
    }  # positions of file,text,speaker_name after split line
    f_extension: str = ".wav"
    cleaner_names: str = None

    def __init__(self, data_dir=None, loaded_mapper_path=None):
        self.lexicon = read_lexicon("./lexicon/pinyin-lexicon-r.txt")

    def create_items(self):
        with open(
            os.path.join(self.data_dir, self.train_f_name), mode="r", encoding="utf-8"
        ) as f:
            for line in f:
                parts = line.strip().split(self.delimiter)
                wav_path = os.path.join(self.data_dir, parts[self.positions["file"]])
                wav_path = (
                    wav_path + self.f_extension
                    if wav_path[-len(self.f_extension) :] != self.f_extension
                    else wav_path
                )
                text = parts[self.positions["text"]]
                speaker_name = parts[self.positions["speaker_name"]]
                self.items.append([text, wav_path, speaker_name])

    def get_one_sample(self, item):
        text, wav_path, speaker_name = item
        audio, rate = sf.read(wav_path, dtype="float32")

        #text_ids = np.asarray(self.text_to_sequence(text), np.int32)
        text = ' '.join(text.split())
        text_ids = np.asarray(wanmei_text_to_sequence(text), np.int32)

        # reshape
        text_ids = text_ids.reshape(text_ids.shape[0]*text_ids.shape[1],)

        utt_id = os.path.basename(wav_path).split(".")[0]

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": str(utt_id),
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def setup_eos_token(self):
        return None # because we do not use this

    def save_pretrained(self, saved_path):
        os.makedirs(saved_path, exist_ok=True)
        self._save_mapper(os.path.join(saved_path, PROCESSOR_FILE_NAME), {})

    def text_to_sequence(self, text):
        # is Chinese characters
        is_translate_pinyin = False
        data = text.split()
        for per_char in data:
            if is_zh(per_char):
                is_translate_pinyin = True
                break

        if is_translate_pinyin is True:
            # chinese character -> pinyin
            phones = []
            pinyins = [
                p[0]
                for p in pinyin(
                    text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
                )
            ]
            for p in pinyins:
                if p in self.lexicon:
                    phones += self.lexicon[p]
                else:
                    phones.append("PAUSE")
            text = " ".join(phones)

        text = ' '.join(text.split())
        text_ids = np.asarray(wanmei_text_to_sequence(text), np.int32)
        
        # reshape
        text_ids = text_ids.reshape(text_ids.shape[0]*text_ids.shape[1],)

        return text_ids

    def inference_text_to_seq(self, text: str):
        return self.symbols_to_ids(self.text_to_ph(text))

    def symbols_to_ids(self, symbols_list: list):
        return [self.symbol_to_id[s] for s in symbols_list]

