import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
Facemodel=load_model('D:\\sales project models two\\models\\FaceDetectorModel.keras')
ImageClassifier=load_model('D:\\sales project models two\\models\\ImageClassifierModel.keras')

def faceDet(imag_dir):
    img = cv2.imread(imag_dir)
    resize = tf.image.resize(img, (256,256))
    yhat = Facemodel.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5:
        return False
    else:
        return True


def ImageClass(imag_dir):
    img = cv2.imread(imag_dir)
    resize = tf.image.resize(img, (256,256))
    yhat = ImageClassifier.predict(np.expand_dims(resize/255, 0))
    if yhat > 0.5:
        return 'Sad'
    else:
        return 'Happy'
