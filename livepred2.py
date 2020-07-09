import keras
import numpy as np
import librosa

class livePredictions:
    def __init__(self, path, file):
        self.path = path
        self.file = file

    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        print(self.file)
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, -1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print( "Prediction is", " ", self.convertclasstoemotion(predictions))

    @staticmethod
    def convertclasstoemotion(pred):
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label



#pred = livePredictions(path='Emotion_Voice_Detection_Model.h5',
 #                      file='/home/avinashchavan/Desktop/test_audio/QR-[2020.04.07]-003844.wav')

pred = livePredictions(path='/home/avinashchavan/Desktop/finalproject/Model_C_f.h5',
                       file='/home/avinashchavan/Desktop/test_audio/angry _2_C_Anup.wav')

pred.load_model()
pred.makepredictions()
