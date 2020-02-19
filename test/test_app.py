#!/usr/bin/python
# coding=utf-8

import pytest
import requests


@pytest.mark.parametrize('endpoint,expected', [('/mnist/4', '{"label": 4, "predicted": 4}'),
                                               ('/mnist/4432', '{"label": 0, "predicted": 0}'),
                                               ('/mnist/10000',
                                                'Invoke using the form /mnist/<index of test image>. Index must be in '
                                                'the range [0:9999]. For example, /mnist/24'),
                                               ('/mnist',
                                                'Not found. Invoke using the form /mnist/<index of test image>. For '
                                                'example, /mnist/24'),
                                               ('/mni',
                                                'This service verifies a model using the MNIST Test data set. Invoke '
                                                'using the form /mnist/<index of test image>. For example, /mnist/24'),
                                               ('/mnist/test',
                                                'Not found. Invoke using the form /mnist/<index of test image>. For '
                                                'example, /mnist/24')])
def test_request_response(mock_port, mock_thread, endpoint, expected):
    """Test that the correct messages are returned for each endpoint."""
    url = f'http://localhost:{mock_port}{endpoint}'
    response = requests.get(url)
    assert response.text == expected


@pytest.mark.parametrize('endpoint,expected', [('/mnist/4', 200),
                                               ('/mnist/4432', 200),
                                               ('/mnist/10000', 404),
                                               ('/mnist', 404),
                                               ('/mni', 400),
                                               ('/mnist/test', 404)])
def test_request_status(mock_port, mock_thread, endpoint, expected):
    """Test that the correct statuses are returned for each endpoint."""
    url = f'http://localhost:{mock_port}{endpoint}'
    response = requests.get(url)
    assert response.status_code == expected
