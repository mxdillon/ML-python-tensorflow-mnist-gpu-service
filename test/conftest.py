#!/usr/bin/python
# coding=utf-8

import pytest
import numpy as np
import falcon

from falcon import testing

from service import MNIST, Intro
from app import number_of_workers


@pytest.fixture(scope="session")
def mnist_class():
    return MNIST()


@pytest.fixture(scope="session")
def json_out(mnist_class):
    return mnist_class.format_payload(index=0, y_pred=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))


@pytest.fixture(scope="session")
def test_image():
    return np.zeros(shape=(28, 28))


@pytest.fixture(scope="session")
def test_client(mnist_class):
    options = {
        'bind': '%s:%s' % ('0.0.0.0', '8080'),
        'workers': str(number_of_workers()),
    }
    client = testing.TestClient(falcon.API(), options)
    client.app.add_route("/mnist/{index:int(min=0)}", mnist_class)
    client.app.add_route("/mnist", Intro())
    return client
