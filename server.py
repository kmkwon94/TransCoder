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
import fastBPE
import torch
import preprocessing.src.code_tokenizer as code_tokenizer
from XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from XLM.src.model import build_model
from XLM.src.utils import AttrDict
from translate import Translator
from threading import Thread

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)
SUPPORTED_LANGUAGES = ['cpp', 'java', 'python']
threads = []
#####################################################
# multi-threads with return value
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, timeout=3):
        Thread.join(self, timeout=3)
        return self._return

class thread_with_trace(ThreadWithReturnValue):
    def __init__(self, *args, **keywords):
        ThreadWithReturnValue.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        ThreadWithReturnValue.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True
#####################################################

#####################################################
#preload model
translator1 = Translator('/TransCoder/checkpoints/model_1.pth')
translator2 = Translator('/TransCoder/checkpoints/model_2.pth')
#####################################################

@app.route('/')
def render_file():
    return render_template('index.html')

@app.route('/healthz', methods=['GET'])
def healthz():
    return "I am alive", 200

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        global threads

        if len(threads) > 3:
            return Response("error : Too many requests", status=429)

        userDirName = str(uuid.uuid4())
        f = request.files['file'] #input_dir

        check_source_value = request.form['source'] #source_lang
        check_target_value = request.form['target'] #target_lang
        input_file = f.read()
        input_file_string = input_file.decode('utf-8')
        try:
            if check_source_value == 'cpp' and check_target_value == 'java' : 
                result = run_model_1(input_file_string, check_source_value, check_target_value, userDirName)
                return render_template('index.html', rawString = result)
        except Exception :
            return Response("There is problem in run model_1 cpp -> java", status = 400)
        try:
            if check_source_value == 'java' : 
                result = run_model_1(input_file_string, check_source_value, check_target_value, userDirName)
                return render_template('index.html', rawString = result)
        except Exception :
            return Response("There is problem in run model_1 java -> python or cpp", status = 400)
        try:
            if check_source_value == 'python' :
                result = run_model_2(input_file_string, check_source_value, check_target_value, userDirName)
                return render_template('index.html', rawString = result)
        except Exception :
            return Response("There is problem in run model_2 python -> java or cpp", status=400)
        try:
            if check_source_value == 'cpp' and check_target_value == 'python' : 
                result = run_model_2(input_file_string, check_source_value, check_target_value, userDirName)
                return render_template('index.html', rawString = result)
        except Exception :
            return Response("There is problem in run model_2 cpp -> python", status=400)
   
def run_model_1(input_dir, src_lang, tgt_lang, user_key):
    try:
        with torch.no_grad():
            #handling multi-threads
            t1 = thread_with_trace(target=translator1.translate, args=(input_dir, src_lang, tgt_lang))
            t1.user_id = user_key
            threads.append(t1)
            while threads[0].user_id!=user_key:
                if threads[0].is_alive():
                    threads[0].join()
            threads[0].start()
            
            output = threads[0].join(timeout=3)
            if threads[0].is_alive():
                threads[0].kill()
                threads.pop(0)
                raise Exception("error model does not work! please try again 30 seconds later")
            threads.pop(0)
        return output
    except Exception as e:
        print(e)
        return Response("error! please try again", status=400)

def run_model_2(input_dir, src_lang, tgt_lang, user_key):
    try:
        with torch.no_grad():
            #handling multi-threads
            t1 = thread_with_trace(target=translator2.translate, args=(input_dir, src_lang, tgt_lang))
            t1.user_id = user_key
            threads.append(t1)
            while threads[0].user_id!=user_key:
                print(str(user_key)+": ", threads[0].user_id)
                if threads[0].is_alive():
                    threads[0].join()
            threads[0].start()
            
            output = threads[0].join(timeout=3)
            if threads[0].is_alive():
                threads[0].kill()
                threads.pop(0)
                raise Exception("error model does not work! please try again 30 seconds later")
            threads.pop(0)
        return output
    except Exception as e:
        print(e)
        return Response("error! please try again", status=400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)