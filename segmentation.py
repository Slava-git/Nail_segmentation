import io
import numpy as np
from PIL import Image
import streamlit as st
from samples.custom.nails import InferenceConfig
from mrcnn.visualize import display_instances
from mrcnn import model as modellib
from samples.custom.nails import DEFAULT_LOGS_DIR 


def load_model():
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=DEFAULT_LOGS_DIR)
    model_path = model_path = 'C:/Users/Viacheslav/Downloads/rcnn_9_epochs.h5'

    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(model, image):
    #image = np.asarray(image)
    # if image.shape[-1] == 4:
    #     image = image[..., :3]

    img_arr = np.array(image)
    results = model.detect([img_arr], verbose=0)
    r = results[0]
    img = display_instances(np.asarray(image), r['rois'], r['masks'], r['class_ids'], 
                                ['BG', 'Nail'], r['scores'], figsize=(5,5), show_bbox=False)

    return img


def main():
    st.title('Image upload demo')
    image = load_image()
    result = st.button('Run on image')
    if result:
        model = load_model()
        predicted_image = predict(model, image)
        st.image(predicted_image, caption='Sunrise in your heart')


if __name__ == '__main__':
    main()