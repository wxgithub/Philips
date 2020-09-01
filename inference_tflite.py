from __future__ import print_function

import numpy as np
#import cv2
import tensorflow as tf
import os, argparse, pathlib
from data import process_image_file
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--weightspath', default='models/', type=str, help='Path to output folder')
parser.add_argument('--metaname', default='model-18540.meta', type=str, help='Name of ckpt meta file')
parser.add_argument('--ckptname', default='model-18540', type=str, help='Name of model ckpts')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path="./covid-19.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32
h = input_details[0]['shape'][1]
w = input_details[0]['shape'][2]

print("== Input details ==")
print("name:", input_details[0]['name'])
print("shape:", input_details[0]['shape'])
print("type:", input_details[0]['dtype'])

print("\n== Output details ==")
print("name:", output_details[0]['name'])
print("shape:", output_details[0]['shape'])
print("type:", output_details[0]['dtype'])

interpreter.allocate_tensors()

x = process_image_file(args.imagepath, args.top_percent, args.input_size)
x = x.astype('float32') / 255.0

inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}
#tensor_index = interpreter.get_input_details()[0]['index']
#input_tensor = interpreter.tensor(tensor_index)()[0]
input_tensor = np.expand_dims(x, axis=0)
print(input_tensor.shape)
interpreter.set_tensor(input_details[0]['index'], input_tensor)

start = timer()
interpreter.invoke()
end = timer()
print('Elapsed time is ', (end-start)*1000, 'ms')
pred = interpreter.get_tensor(output_details[0]['index'])
print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
print('Confidence')
print('Normal: {:.3f}, Pneumonia: {:.3f}, COVID-19: {:.3f}'.format(pred[0][0], pred[0][1], pred[0][2]))

