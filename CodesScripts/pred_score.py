# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import os
import torch
import torch.nn as nn
from torchvision import transforms
import json
import joblib
import copy
import torch.nn.functional as F
import math
import numpy
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import Predictor

from azureml.core.model import Model
from Predictor import ObjectBehaviorModel

from inference_schema.schema_decorators import input_schema
from inference_schema.schema_decorators import output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    print("xxxxyyxxx")
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = Model.get_model_path(model_name='pytorch-BD')
    #path="./azureml-models/pytorch-BD/3/model/amodel.pt"
    print(os.getcwd())
    #model = torch.load(model_path, map_location=lambda storage, loc: storage)
    #print(model_path)
    model = joblib.load(model_path)

input_sample = numpy.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
output_sample = numpy.array([
    5021.509689995557,
    3693.645386402646])


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    #input_data = torch.tensor(json.loads(input_data)['data'])

    # get prediction
    #output = model.inference(input_data)
    #output='well done!'
    classes = ['chicken', 'turkey']
    #softmax = nn.Softmax(dim=1)
    #pred_probs = softmax(output).numpy()[0]
    #index = torch.argmax(output, 1)

    result = {"label": classes[1], "probability": str(0.986)}
    return result
