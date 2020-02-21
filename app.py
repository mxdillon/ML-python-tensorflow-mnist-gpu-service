#!/usr/bin/python
# coding=utf-8

from http.server import BaseHTTPRequestHandler, HTTPServer
from service import MNIST

PORT_NUMBER = 8080


class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        """Handle HTTP GET request.

        If endpoint '/mnist/xxx' where 'xxx' is an integer [0-9999] the result of the prediction will be returned to
        the user. If not, a helpful message will be displayed detailing the correct endpoint."""
        mnist_inference = MNIST()

        path_components = self.path.split("/")
        if path_components[1] == 'mnist':

            try:
                index = int(path_components[2])
                if index < mnist_inference.image_count:
                    body = mnist_inference.run_inference(index)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(bytes(body, "utf8"))
                else:
                    self.send_response(404)
                    self.end_headers()
                    body = f"Invoke using the form /mnist/<index of test image>. Index must be in the range " \
                           f"[0:{mnist_inference.image_count - 1}]. For example, /mnist/24"
                    self.wfile.write(bytes(body, "utf8"))

            except (IndexError, ValueError):
                self.send_response(404)
                self.end_headers()
                body = "Not found. Invoke using the form /mnist/<index of test image>. For example, /mnist/24"
                self.wfile.write(bytes(body, "utf8"))

        else:
            self.send_response(400)
            self.end_headers()
            body = "This service verifies a model using the MNIST Test data set. Invoke using the form /mnist/<index " \
                   "of test image>. For example, /mnist/24"
            self.wfile.write(bytes(body, "utf8"))


if __name__ == "__main__":
    try:
        server = HTTPServer(('', PORT_NUMBER), MyHandler)
        print('Started httpserver on port', PORT_NUMBER)
        server.serve_forever()

    except KeyboardInterrupt:
        server.server_close()
        print('Stopping server')
