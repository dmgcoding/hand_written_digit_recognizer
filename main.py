from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('models/recognize_digits_with_cnn.h5')
    print('model loaded')
except Exception as e:
    print(e)

@app.route('/digit/recog',methods=['POST'])
def digit_recog():
    try:
        file = request.files['img']

        # save image
        file.save('uploads/' + file.filename)

        # open in opencv
        img = cv2.imread('uploads/' + file.filename)

        # resize
        resized_image = cv2.resize(img, (28, 28))

        # overwrite the file
        cv2.imwrite('uploads/'+file.filename, resized_image)
        # lose the depth
        grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # predict
        predicts = model.predict(np.array([grayscale_image]))

        # remove the file
        path = 'uploads/' + file.filename
        if os.path.exists(path):
            # Remove the image file
            os.remove(path)

        return jsonify({
            'msg': str(np.argmax(predicts[0]))
        })
    except Exception as e:
        print(e)
        return jsonify({
            "error": "some error occured"
        })


if __name__ == "__main__":
    app.run(debug=True)