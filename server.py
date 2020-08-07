# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os 
import random
import string
import uuid
import numpy as np
import shutil
import io
import sys
from threading import Thread
from translate import
import argparse
import sys
import fastBPE
import torch
import preprocessing.src.code_tokenizer as code_tokenizer
from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from XLM.src.model import build_model
from XLM.src.utils import AttrDict
from translate import Translator
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
SUPPORTED_LANGUAGES = ['cpp', 'java', 'python']
#####################################################
#preload model
translator = Translator(params)
#####################################################

@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return "I am alive", 200

@app.route('/translate', methods=['POST'])
def translate():
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)