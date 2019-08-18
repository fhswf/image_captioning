import io
import ssl
import urllib.request
from flask import Flask, send_from_directory
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from werkzeug import datastructures
from PIL import Image
from annotator import Annotator

app = Flask(__name__, static_url_path='/ui')
CORS(app)
api = Api(app)

@app.route('/<path:path>')
def send_static(path):
    """Serve static content.
    Static resources for the demo interface are served via the frontend directory.
    
    Parameters
    ----------
    path : str
        Path of requested rersource. 
    """
    return send_from_directory('frontend', path)

class Caption(Resource):
    """REST end point for the captioning service.
    
    Accepts images (via HTTP POST requests) and returns corresponding captions.
    """
    def __init__(self):
        self.annotator = Annotator()

    def post(self):
        """Create caption for an image.
        
        The POST request should include either an image or a url pointing to an image.

        Parameters
        ----------
        image
            Image data send with the request.
        url
            URL of the image to be processed.
        """
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
