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
def get_free_port():
    """Find a free port to run mock server on."""
    s = socket.socket(socket.AF_INET, type=socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port


@pytest.fixture(scope='session')
def thread(get_free_port):
    """Set up mock server and run on a new thread.

    :param get_free_port: port to start server on
    :return: None
    """
    mock_server_port = get_free_port
    mock_server = HTTPServer(('', mock_server_port), MyHandler)
    mock_server_thread = Thread(target=mock_server.serve_forever)
    mock_server_thread.setDaemon(True)
    mock_server_thread.start()
