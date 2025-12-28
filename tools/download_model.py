""" 
Usage: python download.py <model_name>

ex: python download.py yolov8n_seg
"""

from ultralytics import YOLO
import sys

# Load a model
model = YOLO(sys.argv[1] + ".pt")  # load an official model

# Export the model
model.export(format="onnx")