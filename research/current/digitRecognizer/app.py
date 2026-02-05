from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

class Net(nn.Module):
    def __init__(self):
        