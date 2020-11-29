from flask import Flask
from flask import request
from flask import render_template
from flask import url_for
from io import BytesIO
from base64 import b64encode
from werkzeug.utils import secure_filename
import os
from PIL import Image
import utils


app = Flask(__name__, template_folder="./templates")
app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file_object = request.files['img']
        if check_valid_file(file_object):
            image = Image.open(BytesIO(file_object.read())) 
            image = utils.resize_image((224,224), image)
            processed_image = utils.get_model_input(image)
            breed, prob = utils.predict(processed_image)
            
            # for displaying the resized image in the html response page
            image_bytes = BytesIO()
            image.save(image_bytes, format="JPEG")
            uri = "data:image/jpeg;base64," + b64encode(image_bytes.getvalue()).decode('ascii')
    
            if prob < 50:
                result = "Sorry, dog breed could not be determined for this image!"
                breed = "cannot determine"
                prob = "not applicable"
            else:
                result = "Here is your result:"
            return render_template("home.html", uri=uri,
                                         page="prediction", result=result, breed=breed, prob=prob)
        else:
            return render_template("home.html", page="home", 
                                         error_msg="Did you upload a valid file?")
    else:
        return render_template("home.html", page="home", error_msg="")


def check_valid_file(image):
    filename = secure_filename(image.filename) 
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in ['JPG', 'PNG']:
        return True
    else:
        return False

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)