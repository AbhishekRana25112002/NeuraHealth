from django.shortcuts import render,redirect, get_object_or_404
from django.http import HttpResponse
from .models import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
# from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# Create your views here.

def index(request):
    return render(request, 'index.html')

def lung_cancer(request):
    if request.method=="POST":
        image = request.FILES.get('image')
        username = request.POST['name']
        new_image = Image.objects.create(name = str(image), image = image)
        new_image.save()
        
        # 'D:\NeuraHealth' + image_object.image.url
        image_object = Image.objects.get(name = str(image))
        pred_prob = lung_cancer_model(str(image_object.image.url))
        # url = 'D:/NeuraHealth' + image_object.image.url
        context = {
            'name': f"Hii {username}, your report is ready",
            'image' : image_object.image.url,
            'lung_scc' : round(pred_prob[0] * 100, 3),
            'lung_n': round(pred_prob[1] * 100, 3),
            'lung_aca': round(pred_prob[2] * 100, 3)
        }
        Image.objects.all().delete()
        return render(request, 'results.html', context)

    return render(request, 'lung_cancer.html')

def results(request):
    context = {
        'image': 'static/krzysztof-dubiel-hQBIJsBtyBw-unsplash.jpg',
        'name': 'We offer the best predictions available',
    
    }
    return render(request, 'results.html', context)

def pneumonia(request):
    if request.method=="POST":
        image = request.FILES.get('image')
        username = request.POST['name']
        new_image = Image.objects.create(name = str(image), image = image)
        new_image.save()
        
        image_object = Image.objects.get(name = str(image))
        pred_prob = pneumonia_model(str(image_object.image.url))
        probability = "Pneumonia not present"
        if(pred_prob > 0):
            probability = "Pneumonia present"
        context = {
            'name': f"Hii {username}, your report is ready",
            'image' : image_object.image.url,
            'probability' : probability
        }
        Image.objects.all().delete()
        return render(request, 'results2.html', context)

    return render(request, 'pneumonia.html')

def results2(request):
    context = {
        'image': 'static/krzysztof-dubiel-hQBIJsBtyBw-unsplash.jpg',
        'name': 'We offer the best predictions available',
    
    }
    return render(request, 'results2.html', context)

def pneumonia_model(image_path):
    model = load_model("D:/NeuraHealth/Predictors/our_model.h5")
    image_path = 'D:/NeuraHealth' + image_path
    # Load and preprocess a test image
    img = image.load_img(image_path, target_size=(224, 224))
    image_data = image.img_to_array(img)
    image_data = np.expand_dims(image_data, axis=0)
    img_data = preprocess_input(image_data)

    # Make predictions on the test image
    prediction = model.predict(img_data)

    pred_prob = prediction[0][1]
    # Print the prediction result
    print("The probablity of not having pneumonia is: ", prediction[0][0] , "and for having pneumonia is  : " , prediction[0][1])

    return pred_prob


def lung_cancer_model(image_path):

    image_path = 'D:/NeuraHealth' + image_path
    # Function to preprocess the input image
    def preprocess_image(img_path, target_size):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    # Load the trained model
    model = keras.models.load_model("D:\Projects\Deep Learning\LungCancerClassificationTL\lung_cancer_detection_model.h5")

    # Specify the path to the image you want to predict
    # image_path = 'D:\Projects\Deep Learning\LungCancerClassificationTL\lung_colon_image_set\lung_image_sets\lung_aca\lungaca1235.jpeg'  # Replace with the actual path of your image

    # Preprocess the input image
    input_image = preprocess_image(image_path, target_size=(186, 186))

    # Make predictions
    predictions = model.predict(input_image)
    pred_prob = [0, 0, 0]
    print(f"The probability for scc is, {predictions[0][0]}\n for n is {predictions[0][1]}\n and for aca is {predictions[0][2]}")
    pred_prob[0] = predictions[0][0]
    pred_prob[1] = predictions[0][1]
    pred_prob[2] = predictions[0][2]
    # print(predictions)
    # Convert predictions to class labels
    class_labels = ['lung_scc', 'lung_n', 'lung_aca']  # Replace with your actual class labels
    predicted_class_index = np.argmax(predictions)
    # print(predicted_class_index)
    predicted_class_label = class_labels[predicted_class_index]

    # Display the prediction
    print(f'The predicted class is: {predicted_class_label}')
    return pred_prob