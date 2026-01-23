from flask import Flask, request, jsonify
import numpy as np
import time
import os
import cv2



app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def infer():
    end2end_start = time.time()

    image = request.files['image']


















from flask_restful import Api, Resource

app = Flask(__name__)

api = Api(app)


class HelloWorld(Resource):
    def get(self):
        data = {"data": "Hello World"}
        return data


api.add_resource(HelloWorld, '/hello')

if __name__ == '__main__':
    app.run(debug=True)





