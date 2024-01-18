import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

# Function to preprocess the input image
def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the trained model
model = keras.models.load_model("D:\Projects\Deep Learning\LungCancerClassificationTL\lung_cancer_detection_model.h5")

# Specify the path to the image you want to predict
image_path = 'D:\Projects\Deep Learning\LungCancerClassificationTL\lung_colon_image_set\lung_image_sets\lung_aca\lungaca1235.jpeg'  # Replace with the actual path of your image

# Preprocess the input image
input_image = preprocess_image(image_path, target_size=(186, 186))

# Make predictions
predictions = model.predict(input_image)
print(f"The probability for scc is, {predictions[0][0]}\n for n is {predictions[0][1]}\n and for aca is {predictions[0][2]}")
# print(predictions)
# Convert predictions to class labels
class_labels = ['lung_scc', 'lung_n', 'lung_aca']  # Replace with your actual class labels
predicted_class_index = np.argmax(predictions)
# print(predicted_class_index)
predicted_class_label = class_labels[predicted_class_index]

# Display the prediction
print(f'The predicted class is: {predicted_class_label}')
