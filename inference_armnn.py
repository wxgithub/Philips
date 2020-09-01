import PIL
from PIL import Image
import pyarmnn as ann
import numpy as np
#import cv2
from data import process_image_file
from timeit import default_timer as timer

print(ann.ARMNN_VERSION)

import os, argparse, pathlib

parser = argparse.ArgumentParser(description='COVID-Net Inference')
parser.add_argument('--modelname', default='./covid-19.tflite', type=str, help='Path to output folder')
parser.add_argument('--imagepath', default='assets/ex-covid.jpeg', type=str, help='Full path to image to be inferenced')
parser.add_argument('--in_tensorname', default='input_1:0', type=str, help='Name of input tensor to graph')
parser.add_argument('--out_tensorname', default='norm_dense_1/Softmax:0', type=str, help='Name of output tensor from graph')
parser.add_argument('--input_size', default=480, type=int, help='Size of input (ex: if 480x480, --input_size 480)')
parser.add_argument('--top_percent', default=0.08, type=float, help='Percent top crop from top of image')

args = parser.parse_args()

# ONNX, Caffe and TF parsers also exist.
parser = ann.ITfLiteParser()  
network = parser.CreateNetworkFromBinaryFile(args.modelname)

graph_id = 0
input_names = parser.GetSubgraphInputTensorNames(graph_id)
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_tensor_id = input_binding_info[0]
input_tensor_info = input_binding_info[1]
print(input_tensor_id)
print(input_tensor_info)

# Create a runtime object that will perform inference.
options = ann.CreationOptions()
runtime = ann.IRuntime(options)

# Backend choices earlier in the list have higher preference.
#preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('GpuAcc'), ann.BackendId('CpuRef')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

# Load the optimized network into the runtime.
net_id, _ = runtime.LoadNetwork(opt_network)

print(args.imagepath)
#image = cv2.imread(args.imagepath)
image = process_image_file(args.imagepath, args.top_percent, args.input_size)
image = image.astype('float32') / 255.0

input_tensors = ann.make_input_tensors([input_binding_info], [image])

# Get output binding information for an output layer by using the layer name.
output_names = parser.GetSubgraphOutputTensorNames(graph_id)
output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
output_tensors = ann.make_output_tensors([output_binding_info])

start = timer()
runtime.EnqueueWorkload(0, input_tensors, output_tensors)
end = timer()
print("Elapsed time is ", (end - start) * 1000, "ms")
results = ann.workload_tensors_to_ndarray(output_tensors)
print(results[0])
print(output_tensors[0][1])
