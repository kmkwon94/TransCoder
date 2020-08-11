# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, send_file
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
translator1 = Translator('model_1.pth')
translator2 = Translator('model_2.pth')
#####################################################

@app.route('/')
def render_file():
    return render_template('upload.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return "I am alive", 200

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        userDirName = str(uuid.uuid4())
        f = request.files['file'] #input_dir
        check_source_value = request.form['source'] #source_lang
        check_target_value = request.form['target'] #target_lang
        #model_path
        input_file = open(f, 'r')
        input_file_string = input_file.read()
        input_file.close()
        if check_source_value == 'cpp' and check_target_value == 'java' : 
            result = run_model_1(input_file_string, check_source_value, check_target_value, userDirName)
            return render_template('index.html', rawString = result)
        if check_source_value == 'java' : 
            result = run_model_1(input_file_string, check_source_value, check_target_value, userDirName)
            return render_template('index.html', rawString = result)
        if check_source_value == 'python' : 
            result = run_model_2(input_file_string, check_source_value, check_target_value, userDirName)
            return render_template('index.html', rawString = result)
        if check_source_value == 'cpp' and check_target_value == 'python' : 
            result = run_model_2(input_file_string, check_source_value, check_target_value, userDirName)
            return render_template('index.html', rawString = result)
    #model_1
    #c++ -> java 
    #java -> c++
    #java -> python

    #model_2
    #C++ -> python 
    #python -> c++
    #python -> java
def run_model_1(input_dir, src_lang, tgt_lang, userDirName):
    with torch.no_grad():
        output = translator1.translate(input_dir, src_lang, tgt_lang, n = 1, beam_size=1)
    #os.mkdir('./static/' + userDirName)
    #result = makeOutputFile(tgt_lang, output, userDirName)
    return output

def run_model_2(input_dir, src_lang, tgt_lang, userDirName):
    with torch.no_grad():
        output = translator2.translate(input_dir, src_lang, tgt_lang, n = 1, beam_size=1)
    #os.mkdir('./static/' + userDirName)
    #result = makeOutputFile(tgt_lang, output, userDirName)
    return output
'''
def makeOutputFile(tgt_lang, output, userDirName):
    if tgt_lang == 'java':
        path = './static/' + userDirName + '/result.java'
        f = open(path, 'w')
        for out in output:
            print("=" * 20)
            print(out)
            f.write(out)
        f.close()
        
        with open(path, 'rb') as f:
            data = f.read()
        result = io.BytesIO(data)
        remove(userDirName)
        return result
    elif tgt_lang == 'cpp':
        path = './static/' + userDirName + '/result.cpp'
        f = open(path, 'w')
        for out in output:
            print("=" * 20)
            print(out)
            f.write(out)
        f.close()

        with open(path, 'rb') as f:
            data = f.read()
        result = io.BytesIO(data)
        remove(userDirName)
        return result
    else:
        path = './static/' + userDirName + '/result.py'
        f = open(path, 'w')
        for out in output:
            print("=" * 20)
            print(out)
            f.write(out)
        f.close()
        
        with open(path, 'rb') as f:
            data = f.read()
        result = io.BytesIO(data)
        remove(userDirName)
        return result

def remove(user_key):
    #remove_input_dir = '/ganilla/upload/' + user_key 
    remove_dir = './static/' + user_key 
    print("Now start to remove file")
    print("user key is " + user_key)
    #print("Input path " + remove_input_dir)
    print("Output path " + remove_dir)

    #output path를 삭제하는 try 문
    try:
        if os.path.isdir(remove_dir):
            shutil.rmtree(remove_dir)
            print("Delete " + remove_dir + " is completed")
    except Exception as e:
        print(e)
        print("Delete" + remove_dir + " is failed")
    
    #input path를 삭제하는 try 문
    try:
        if os.path.isdir(remove_input_dir):
            shutil.rmtree(remove_input_dir)
            print("Delete" + remove_input_dir + " is completed")
    except Exception as e:
        print("Delete" + remove_input_dir + " is failed")
       
    return print("All of delete process is completed!")
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)