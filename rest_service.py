from flask import Flask, send_from_directory
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

from werkzeug import datastructures
from PIL import Image
from annotator import Annotator
import urllib.request
import io
import ssl

app = Flask(__name__, static_url_path='/ui')
CORS(app)
api = Api(app)

@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('frontend', path)

class Caption(Resource):
    def __init__(self):
        self.annotator = Annotator()

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("image", type=datastructures.FileStorage, location='files')
        parser.add_argument("url")
        args = parser.parse_args()
        if args.image:
            image = Image.open(args.image)
        else:
            ssl._create_default_https_context = ssl._create_unverified_context
            with urllib.request.urlopen(args.url) as url:
                f = io.BytesIO(url.read())
            
            image = Image.open(f)
            
        annotation = self.annotator.annotate(image)
        data = { 
            "annotation": annotation,
        }
        return data, 201

api.add_resource(Caption, "/caption")

if __name__ == "__main__":
    app.run(debug=True)
