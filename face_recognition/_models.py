# -*- coding: utf-8 -*-
"""Locate the pre-trained dlib model files without importing
`face_recognition_models`.

The published `face_recognition_models` package imports `pkg_resources` at
import time, and setuptools>=82 no longer ships `pkg_resources`. We only need
the paths of its data files, so we find the installed package with
importlib and build the paths ourselves.
"""

import importlib.util
import os

_PACKAGE = "face_recognition_models"

_INSTALL_HINT = (
    "The pre-trained models were not found. Install them with:\n\n"
    "    pip install face_recognition_models\n"
)


def _models_dir():
    try:
        spec = importlib.util.find_spec(_PACKAGE)
    except ValueError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(_INSTALL_HINT)
    for location in spec.submodule_search_locations:
        candidate = os.path.join(location, "models")
        if os.path.isdir(candidate):
            return candidate
    raise ImportError(_INSTALL_HINT)


def _model_path(filename):
    path = os.path.join(_models_dir(), filename)
    if not os.path.isfile(path):
        raise ImportError("Model file is missing: {}\n{}".format(path, _INSTALL_HINT))
    return path


def pose_predictor_model_location():
    return _model_path("shape_predictor_68_face_landmarks.dat")


def pose_predictor_five_point_model_location():
    return _model_path("shape_predictor_5_face_landmarks.dat")


def face_recognition_model_location():
    return _model_path("dlib_face_recognition_resnet_model_v1.dat")


def cnn_face_detector_model_location():
    return _model_path("mmod_human_face_detector.dat")
