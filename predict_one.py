from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
from tqdm import tqdm
import librosa


def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'Melspectrogram':Melspectrogram,
                        'Normalization2D':Normalization2D})

    #wav_paths gets files for class labels
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])

    #single file from args
    wav_fn = args.fn
    classes = sorted(os.listdir(args.src_dir))

    try:
        rate, wav = downsample_mono(wav_fn, args.sr)
    except Exception:
        print("failed (load): " + wav_fn.split('/')[-1])

    #trim the silence continue of error is encountered
    try:
        clean_wav, index = librosa.effects.trim(wav, top_db=60)
    except Exception:
        print("failed (trim): " + wav_fn.split('/')[-1])

    #calculate stetp size in samples
    step = int(args.sr*args.dt)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(1,-1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(1,step), dtype=np.float32)
            tmp[:,:sample.shape[1]] = sample.flatten()
            sample = tmp
        batch.append(sample)

    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    print(y_pred)
    y_mean = np.mean(y_pred, axis=0)
    print(y_mean)
    y_pred = np.argmax(y_mean)
    real_class = os.path.dirname(wav_fn).split('/')[-1]
    print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='models/lstm.h5',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='y_pred',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    parser.add_argument('--fn', type=str, default=None, help='path to test file')
    args, _ = parser.parse_known_args()

    make_prediction(args)
