#!/usr/bin/python
# coding=utf-8

import os
import sys
import json
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import time


class MNIST(object):
    def __init__(self):
        """Load the MNIST test dataset (10000 images). Load onnx model and start inference session."""

        start = time.time()
        mnist = tf.keras.datasets.mnist
        (_, _), (x_test, self.y_test) = mnist.load_data()
        self.x_test = x_test / 255.0
        self.image_count = x_test.shape[0]
        end = time.time()
        print("Loading time: {0:f} secs".format(end - start))

        # Load the ONNX model and check the model is well formed
        if not os.path.exists("model.onnx"):
            sys.exit("There needs to be a model located at 'model.onnx'. Tests will fail if this is not the case.")
        self.model = onnx.load("model.onnx")
        onnx.checker.check_model(self.model)

        # Start inference session
        rt.set_default_logger_severity(0)
        self.sess = rt.InferenceSession("model.onnx")
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def prepare_x_test(image_in: np.ndarray) -> np.ndarray:
        """Format an MNIST image so that it can be used for inference in onnx runtime.

        :param image_in: 2-dim numpy array that will be converted into a 4-dim array
        :type image_in: np.ndarray
        :return: 4-dim array with the first (onnxruntime specific) and last dimensions (batchsize=1) as empty
        :rtype: np.ndarray
        """
        test_image = np.asarray(image_in, dtype='float32')
        test_image = np.expand_dims(test_image, axis=2)
        return np.expand_dims(test_image, axis=0)

    def format_payload(self, index: int, y_pred: np.ndarray) -> json:
        """Format prediction results into a json to be returned to user.

        :param index: int 0-9999 indicating which test image will be processed by the model
        :type index: int
        :param predicted: 10-dim array containing probability distribution over the 10 classes
        :type predicted: int
        :return: json with structure {"label": int, "predicted": int}
        :rtype: json
        """
        payload = {}
        payload["label"] = int(self.y_test[index])
        predicted = int(np.argmax(y_pred))
        payload["predicted"] = predicted
        return json.dumps(payload)

    def run_inference(self, index: int) -> json:
        """Handle HTTP GET request.

        :param index: int 0-9999 indicating which test image will be processed by the model. Defined by user as part of
        the HTTP GET request
        :type index: int
        :return: json containing the predicted label and true label
        :rtype: json
        """
        test_image = self.prepare_x_test(image_in=self.x_test[index, :, :])
        y_pred = self.sess.run(None, {self.input_name: test_image})[0]
        return self.format_payload(index=index, y_pred=y_pred)
