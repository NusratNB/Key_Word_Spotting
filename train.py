import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
import sys
sys.path +=[os.getcwd()]
import tensorflow as tf
import librosa
import numpy as np
import features as features_lib
import scipy.io.wavfile as wavreader
import params as model_params
from tensorflow.keras import Model, layers
import tc_resnet14se
from tensorflow import keras
# from dataclasses import dataclass
import pandas  as pd
from sklearn.metrics import  auc
import random
import scipy.io.wavfile as wavreader
import soundfile as sf
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs,  batch_size=8,  cl_numbers=7,
                 shuffle=True, parameters=None):
        'Initialization'
#         self.dim = dim
        self.batch_size = batch_size        
        self.list_IDs = list_IDs
      #  self.n_channels = n_channels
        self.shuffle = shuffle
        self.cl_numbers = cl_numbers
        self.on_epoch_end()
        self.params = parameters

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        datalength = 15600# int(np.ceil((self.params.patch_window_seconds + self.params.stft_window_seconds - self.params.stft_hop_seconds)* self.params.sample_rate))
#         print("#### {} ######".format(datalength))

        X = np.zeros((self.batch_size, datalength))
#         X = []
        y = np.zeros((self.batch_size, self.cl_numbers))
        

        # Generate data
        for i in range(len(list_IDs_temp)):
            data_np = 0
            
            ID = list_IDs_temp[i] 
            id_split = ID.split(" ")
            cl_ind = id_split[1]
            path_to_audio = id_split[0]

#                 sr_sc, audio_sc  = wavreader.read(path_to_audio)
            try:
                wav_data, sr = sf.read(path_to_audio, dtype=np.int16)
            except:
                print("Error ", path_to_audio)
                continue
#             audio_data = wav_data / 32768.0
            audio_data, sr_lib = librosa.core.load(path_to_audio, sr=16000)
            data_np = audio_data# data.to_numpy().transpose()
            
            if data_np.shape[0] < datalength:
                zeros = np.zeros(datalength-data_np.shape[0])
                final_arr = np.concatenate((data_np, zeros))
            elif data_np.shape[0]>datalength:
                final_arr = data_np[0:datalength]
            else:
#                 diff = audio_data.shape[0] - datalength
                final_arr = data_np
            X[i,] = final_arr
#             X = X.astype('float32')
#             X.append(data_np)
            

            y[i,int(cl_ind)] = 1
#         print("\n#######")
#         print("\n", X.shape)
#         print("\n#######")
        return X, y

 
            
#         print("############")
#         print(X.shape)
#         X_array = np.asarray(X)
       

def preprocess_data(images):
    images = (images - 127.00) / 128.00
    return images

def AUROC():
    return tf.keras.metrics.AUC()

def tc_model(params):
    waveform = layers.Input(shape=(15600,), dtype=tf.float32)
#     print("Input shape: ",waveform.shape)
    
#     waveform_padded = features_lib.pad_waveform(waveform, params)
#     waveform_padded = tf.cast(waveform_padded, tf.float32)
    log_mel_spectrogram, features = features_lib.waveform_to_log_mel_spectrogram_patches(
      waveform)
    print("Input shape: ",waveform.shape)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
#     net = layers.Reshape((params.patch_frames, params.patch_bands), input_shape=(None, params.patch_frames, params.patch_bands))(mfcc)
    out = tc_resnet14se.get_tc_recnet_14_se(mfcc, 7, 1.5)
    tc_resnetse = Model(inputs=waveform, outputs=out)
    return tc_resnetse

main_path =  "/Projects/keepin_data/keepin_vt_data/DataNotNormalized/"
train_txt = main_path + 'train.txt'
dev_txt = main_path + 'dev.txt'

val_txt = open(dev_txt, "r+")
tr_txt = open(train_txt, "r+")
tr_patch_list = tr_txt.readlines()
val_patch_list = val_txt.readlines()
random.shuffle(tr_patch_list)
random.shuffle(val_patch_list)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/Projects/keepin_data/TC_ResNet_SE/tc_resnet14_se_prprop_full/2604_keepin_data/model.{epoch:02d}-{val_loss:.2f}.h5', monitor="val_loss", save_best_only=True)

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/Projects/keepin_data/TC_ResNet_SE/tc_resnet14_se_prprop_full/model_tc_resnet_0110_64melbins/', write_images=True)
sr=16000
params = model_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
tr_generator = DataGenerator(tr_patch_list, 2*32, parameters=params)
vl_generator = DataGenerator(val_patch_list, 2*32, parameters=params)

# model = tc_resnet14_se_up.get_tc_recnet_14_se((100, 40), 11, 1.5)

model = tc_model(params)
model.summary()
lr_schedule =tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=100,
    decay_rate=0.9)
#, callbacks=[checkpoint_callback, tensorboard_callback]
opt = tf.keras.optimizers.Adam(lr_schedule)
# loss = tf.keras.losses.CategoricalCrossentropy
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.fit_generator(tr_generator, epochs=50, validation_data=vl_generator, callbacks=[checkpoint_callback])

