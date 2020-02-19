# ML-python-tensorflow-mnist-gpu-service

Quickstart project for executing a MNIST classifier using TensorFlow on a GPU.

* Running `pip install requirements.txt` and then `python app.py` will start the app on localhost where the user can
 send GET requests to perform inference. 

* __NOTE:__ you will need to be running this service on a GPU for the model to load.

* eg `.../mnist/{index}` where `index` is in the range [0:9999]

* __NOTE:__ this quickstart has no onnx model included. This means the tests will fail on the first build.

* This quickstart has been written using the standard `http.server` python library as there are currently issues with
 using gunicorn with onnx models on a GPU.