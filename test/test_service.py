#!/usr/bin/python
# coding=utf-8

import json


def test_download_mnist(mnist_class):
    assert mnist_class.image_count == 10000


def test_download_mnist_xshape(mnist_class):
    assert mnist_class.x_test.shape == (10000, 28, 28)


def test_download_mnist_yshape(mnist_class):
    assert mnist_class.y_test.shape == (10000,)


def test_model_loaded(mnist_class):
    assert mnist_class.model is not None


def test_session_initiated(mnist_class):
    assert mnist_class.sess is not None


def test_input_name(mnist_class):
    assert type(mnist_class.input_name) == str


def test_prepare_x_test(mnist_class, test_image):
    assert mnist_class.prepare_x_test(image_in=test_image).shape == (1, 28, 28, 1)


def test_format_payload_keys(mnist_class, json_out):
    assert list(json.loads(json_out).keys()) == ["label", "predicted"]


def test_format_payload_values(mnist_class, json_out):
    assert list(json.loads(json_out).values()) == [7, 3]
