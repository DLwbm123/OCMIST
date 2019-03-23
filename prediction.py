import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import time
# import tables

from keras import backend as K
from keras.models import load_model
K.set_image_dim_ordering('tf')

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                            weighted_dice_coefficient_loss, weighted_dice_coefficient)

import nibabel as nib

def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return nib.Nifti1Image(data, affine)

def predict_one(model, data, affine):
    # 需要把+1映射到316.685，-1映射到-316.685
    start = time.time()
    print('predict')
    prediction = model.predict(np.asarray([data]))
    print('predicted', time.time() - start)
    return prediction
    # prediction_image = prediction_to_image(prediction, affine, labels = (1,), label_map = True)
    # prediction_image.to_filename(os.path.join('.', "test.nii.gz"))

def load_trained_model(filename):
    # prepare:
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    custom_objects["InstanceNormalization"] = InstanceNormalization
    start = time.time()
    print('load model')
    model = load_model(filename + '.h5', custom_objects=custom_objects)
    print('model loaded', time.time() - start)
    return model
