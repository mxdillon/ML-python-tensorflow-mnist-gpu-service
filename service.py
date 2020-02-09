#!/usr/bin/python

import os
import sys
import falcon
import json
import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import time


PORT_NUMBER = 8080


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
            sys.exit("There needs to be a model located at '.model.onnx'. Tests will fail if this is not the case.")
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

    def on_get(self, req, resp, index: int):
        """Handle HTTP GET request.

        :param req: HTTP request
        :type req: req
        :param resp: HTTP response
        :type resp: resp
        :param index: int 0-9999 indicating which test image will be processed by the model. Defined by user as part of
        the HTTP GET request
        :type index: int
        :raises falcon.HTTPBadRequest: If user passes integer outside acceptable range, bad request raised
        """
        if index < self.image_count:
            test_image = self.prepare_x_test(image_in=self.x_test[index, :, :])
            y_pred = self.sess.run(None, {self.input_name: test_image})[0]
            resp.body = self.format_payload(index=index, y_pred=y_pred)
            resp.status = falcon.HTTP_200
        else:
            raise falcon.HTTPBadRequest(
                "Index Out of Range. ",
                "The requested index must be between 0 and {:d}, inclusive.".format(
                    self.image_count - 1
                ),
            )


class Intro(object):
    def on_get(self, req, resp):
        """Handle HTTP GET request when no MNIST test image is specified.

        :param req: HTTP GET request
        :type req: req
        :param resp: HTTP response
        :type resp: resp
        """
        resp.body = '{"message": \
                    "This service verifies a model using the MNIST Test data set. Invoke using the form \
                    /mnist/<index of test image>. For example, /mnist/24"}'
        resp.status = falcon.HTTP_200
