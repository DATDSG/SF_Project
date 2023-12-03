from timeit import default_timer as timer
from PIL import Image
import tensorflow as tf
import numpy as np
def image_transformer(image_paths):
    images = []
    or_images = []
    for image in image_paths:
        img = Image.open(image)
        or_images.append(img)
        tens_img = tf.convert_to_tensor(img)
        rimg = tf.image.resize(tens_img, [32, 32]).numpy()
        f_img = rimg / 255.0
        images.append(f_img)
    return np.array(images), or_images


def pred_and_plot_image(
    model: tf.keras.Sequential,
    class_names,
    image_paths,
    transformer=image_transformer):


    np_images, or_images = image_transformer(image_paths)

    start_time = timer()

    pred_probs = model.predict(np_images)

    end_time = timer()

    t_time = end_time - start_time

    y_pred_label = np.argmax(pred_probs, axis=-1)
    throughput = len(or_images) / t_time
    pred_classes = [class_names[i][2:] for i in y_pred_label]

    main_list = []
    # Plot image with predicted label and probability
    for i in range(len(pred_classes)):
        data_l = [pred_probs[i]]
        score = tf.nn.softmax(data_l)
        score = np.max(score)
        main_list.append(score)

    return main_list, pred_classes, throughput, t_time