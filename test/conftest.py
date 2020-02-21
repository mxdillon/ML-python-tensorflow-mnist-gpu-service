#!/usr/bin/python
# # coding=utf-8


import numpy as np
import pytest
import socket

from http.server import HTTPServer
from threading import Thread

from app import MyHandler
from service import MNIST


@pytest.fixture(scope="session")
def mnist_class():
    return MNIST()


@pytest.fixture(scope='session')
def port_number():
    return 5050


@pytest.fixture(scope="session")
def json_out(mnist_class):
    return mnist_class.format_payload(index=0, y_pred=np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]))


@pytest.fixture(scope="session")
def test_image():
    return np.zeros(shape=(28, 28))


@pytest.fixture(scope='session')
def mock_port():
    """Find a free port to run mock server on."""
    test_socket = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    test_socket.bind(('localhost', 0))
    _, port = test_socket.getsockname()
    test_socket.close()
    return port


@pytest.fixture(scope='session')
def mock_thread(mock_port):
    """Set up mock server and run on a new thread.

    :param get_free_port: port to start server on
    :return: None
    """
    mock_server = HTTPServer(('', mock_port), MyHandler)
    mock_thread = Thread(target=mock_server.serve_forever)
    mock_thread.setDaemon(True)
    mock_thread.start()
