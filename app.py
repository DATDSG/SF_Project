import os
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model

from essentials import pred_and_plot_image

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html', result=None)


remov_directory = []

@app.route('/predict', methods=['POST'])
def predict(remov_directory=remov_directory):
    # setting path to the model
    model_path = Path("model/gfgModel (1).h5") # methana modelname eka wenas karanna

    # Define the class names for predictions
    class_names = ['0: airplane', '1: automobile', '2: bird', '3: cat', '4: deer', '5: dog', '6: frog', '7: horse', '8: ship', '9: truck']

    # Loading the deep learning model
    model = load_model(model_path)  # Replaced the model path

    # Get the uploaded image file
    images = request.files.getlist("image")

    if not os.path.exists('static'):
        os.mkdir('static')

    ac_images_list = []

    for image in images:
        img_path = "static/" + image.filename
        remov_directory.append(img_path)
        image.save(img_path)
        ac_images_list.append(img_path)

    main_lists_of_probs, y_pred_label, t_put, t_time = pred_and_plot_image(model=model, class_names=class_names,
                                                     image_paths=ac_images_list)
    data = {"image_name":ac_images_list,
            "Probs":main_lists_of_probs,
            "Labels": y_pred_label}

    latency = 1 / t_put
    latency = round(latency, 4)
    t_put = round(t_put, 4)
    t_time = round(t_time, 4)
    return render_template('testingoutput.html',
                           data=data,
                           lenght=t_time,
                           tput=t_put,
                           latency=latency)


@app.route('/goback')
def go_back(remov_directory=remov_directory):
    for i in remov_directory:
        os.remove(i)
    del remov_directory[:]
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
