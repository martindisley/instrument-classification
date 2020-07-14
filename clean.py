import matplotlib.pyplot as plt
from scipy.io import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono, load
from tqdm import tqdm
import numpy
import audioread
import librosa


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/20),
                       min_periods=1,
                       center=True).max()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean


def downsample_mono(path, sr):
    #print(path)
    #rate, wav = wavfile.read(path)
    wav, rate = load(path)
    #wav = wav.astype(np.float32, order='F')
    #wav = wav * 32768
    try:
        tmp = wav.shape[1]
        wav = to_mono(wav.T)
    except:
        pass

    try:
        wav = resample(wav, rate, sr)
    except:
        pass
    #wav = wav.astype(np.int16)
    return sr, wav


#new def for loading audio to avoid using LibROSA.load
def load(filepath, sr=16000, mono=True, offset=0.0, duration=None, dtype=np.float32):

    y = []
    with audioread.audio_open(os.path.realpath(filepath)) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        #set the start time
        s_start = int(np.round(sr_native * offset)) * n_channels
        #set end time: if it a duration is not given leave it infinite, otherwise calculate it
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration))
                               * n_channels)
        #frame index counter
        n = 0

        for block in input_file:
            #convert block into frame of dtype given
            frame = librosa.util.buf_to_float(block, dtype=dtype)
            n_prev = n
            #increment the frame
            n = n + len(frame)
            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[:s_end - n_prev]
            if n_prev <= s_start <= n:
                frame = frame[(s_start - n_prev):]
            #append frames to output
            y.append(frame)

    if y:
        #combine all frames
        y = np.concatenate(y)
        #check the original number of channels
        if n_channels > 1:
            #if greater than 1 split the array back into the original channels
            y = y.reshape((-1, 2)).T
            #if mono is switched convert channels to mono
            if mono:
                y = librosa.to_mono(y)
        #if samplerate is specified
        if sr is not None:
            y = librosa.resample(y, sr_native, sr)
        else:
            #do not change the samplearate but pass back the native samplerate
            sr = sr_native

    y = np.ascontiguousarray(y, dtype=dtype)
    return (y, sr)


def save_sample(sample, rate, target_dir, fn, ix):
    fn = fn.split('.wav')[0]
    dst_path = os.path.join(target_dir.split('.')[0], fn+'_{}.wav'.format(str(ix)))
    #if os.path.exists(dst_path):
        #return
    wavfile.write(dst_path, rate, sample)


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def split_wavs(args):
    src_root = args.src_root
    dst_root = args.dst_root
    dt = args.delta_time

    wav_paths = glob('{}/**'.format(src_root), recursive=True)
    wav_path = [x for x in wav_paths if args.fn in x]
    dirs = os.listdir(src_root)
    check_dir(dst_root)
    classes = os.listdir(src_root)
    for _cls in classes:
        target_dir = os.path.join(dst_root, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(src_root, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)

            try:
                rate, wav = downsample_mono(src_fn, args.sr)
            except Exception:
                print("failed (load): " + src_fn.split('/')[-1])
                continue

            try:
                wav_trim, index = librosa.effects.trim(wav, top_db=60)
            except Exception:
                print("failed (trim): " + src_fn.split('/')[-1])
                continue

            #save_sample(wav_trim, rate, target_dir, fn, 'test-trim')

            #mask, y_mean = envelope(wav, rate, threshold=args.threshold)
            #wav = wav[mask]
            delta_sample = int(dt*rate)

            # cleaned audio is less than a single sample
            # pad with zeros to delta_sample size
            if wav.shape[0] < delta_sample:
                sample = np.zeros(shape=(delta_sample,), dtype=np.int16)
                sample[:wav_trim.shape[0]] = wav
                save_sample(sample, rate, target_dir, fn, 0)

            # step through audio and save every delta_sample
            # discard the ending audio if it is too short
            else:
                trunc = wav_trim.shape[0] % delta_sample
                for cnt, i in enumerate(np.arange(0, wav_trim.shape[0]-trunc, delta_sample)):
                    start = int(i)
                    stop = int(i + delta_sample)
                    sample = wav_trim[start:stop]
                    save_sample(sample, rate, target_dir, fn, cnt)


def test_threshold(args):
    #src_root = args.src_root
    #wav_paths = glob('{}/**'.format(src_root), recursive=True)
    #wav_path = [x for x in wav_paths if args.fn in x]
    wav_path = '/home/martin/Delic-Dev/Audio-Classification/Vocal.wav'
    #if len(wav_path) != 1:
        #print('audio file not found for sub-string: {}'.format(args.fn))
        #return

    #rate, wav = downsample_mono(wav_path[0], args.sr)
    wav, rate = load(wav_path)
    #REPLACE EVELOP WITH LIBROSA TRIM
    mask, env = envelope(wav, rate, threshold=args.threshold)

    plt.style.use('ggplot')
    plt.title('Signal Envelope, Threshold = {}'.format(str(args.threshold)))
    plt.plot(wav[np.logical_not(mask)], color='r', label='remove')
    plt.plot(wav[mask], color='c', label='keep')
    plt.plot(env, color='m', label='envelope')
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Cleaning audio data')
    parser.add_argument('--src_root', type=str, default='wavfiles-a',
                        help='directory of audio files in total duration')
    parser.add_argument('--dst_root', type=str, default='clean-a',
                        help='directory to put audio files split by delta_time')
    parser.add_argument('--delta_time', '-dt', type=float, default=1.0,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='rate to downsample audio')
    parser.add_argument('--fn', type=str, default='3a3d0279',
                        help='file to plot over time to check magnitude')
    parser.add_argument('--threshold', type=str, default=0.01,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    #test_threshold(args)
    split_wavs(args)
