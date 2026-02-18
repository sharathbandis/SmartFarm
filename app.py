from flask import Flask, render_template, request, make_response
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
import joblib

# Import from the utils folder
from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic

app = Flask(__name__)

# --- 1. Load Models & Data ---

# Load Crop Model
try:
    crop_recommendation_model = joblib.load('RandomForest.pkl')
except Exception as e:
    print(f"Error loading RandomForest: {e}")

# Load Disease Model
disease_model = ResNet9(3, 38)
try:
    checkpoint = torch.load('Plant_Disease_Model.pth', map_location=torch.device('cpu'))
    disease_model.load_state_dict(checkpoint)
    disease_model.eval()
except Exception as e:
    print(f"Error loading Disease Model: {e}")

# Load Fertilizer CSV
try:
    fert_data = pd.read_csv('fertilizer.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")

# --- 2. Helper Functions ---

def weather_fetch(city_name):
    """
    Fetches real-time temperature and humidity from OpenWeatherMap.
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    
    try:
        response = requests.get(complete_url)
        x = response.json()
        if x["cod"] != "404":
            y = x["main"]
            temperature = round((y["temp"] - 273.15), 2) # Convert Kelvin to Celsius
            humidity = y["humidity"]
            return temperature, humidity
        else:
            return None
    except:
        return None

def predict_image(img, model=disease_model):
    """
    Preprocesses image and predicts disease.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(img_u)
        # The classes list must match the training order exactly.
        # Assuming PlantVillage 38 classes standard order:
        classes = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        predicted_class = classes[prediction.argmax()]
    return predicted_class

# --- 3. Routes ---

@app.route('/')
def home():
    return render_template('index.html', title="Home")

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html', title="Crop Recommendation")

@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        try:
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorous'])
            K = int(request.form['pottasium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            city = request.form.get("city")

            # Real-Time Logic: Get weather from API
            if weather_fetch(city) != None:
                temperature, humidity = weather_fetch(city)
            else:
                # Fallback if API fails
                temperature = 25.0 
                humidity = 50.0

            # Prediction
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title="Crop Result")
        except Exception as e:
            print(f"Crop Error: {e}")
            return render_template('try_again.html', title="Error")

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title="Fertilizer Advice")

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    try:
        crop_name = str(request.form['cropname'])
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        df = pd.read_csv('fertilizer.csv')
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        
        if max_value == "N":
            key = 'NHigh' if n < 0 else "Nlow"
        elif max_value == "P":
            key = 'PHigh' if p < 0 else "Plow"
        else:
            key = 'KHigh' if k < 0 else "Klow"

        response = Markup(str(fertilizer_dic[key]))
        return render_template('fertilizer-result.html', recommendation=response, title="Fertilizer Advice")
    except Exception as e:
        print(f"Fertilizer Error: {e}")
        return render_template('try_again.html', title="Error")

@app.route('/disease')
def disease_prediction():
    return render_template('disease.html', title="Disease Detection")

@app.route('/disease-predict', methods=['POST'])
def disease_predict():
    try:
        if 'file' not in request.files:
            return render_template('try_again.html')
        file = request.files.get('file')
        if not file:
            return render_template('try_again.html')

        img = file.read()
        prediction = predict_image(img)
        
        # Get description logic
        desc = disease_dic.get(prediction, "No description available.")
        return render_template('disease-result.html', prediction=prediction, title="Disease Result")
    except Exception as e:
        print(f"Disease Error: {e}")
        return render_template('try_again.html')

# --- Download Routes (For Report Button) ---
@app.route('/download_fertilizer', methods=['POST'])
def download_f():
    text = request.form['fileData']
    response = make_response(text)
    response.headers["Content-Disposition"] = "attachment; filename=Fertilizer_Report.txt"
    return response

@app.route('/download_disease', methods=['POST'])
def download_d():
    text = request.form['fileData']
    response = make_response(text)
    response.headers["Content-Disposition"] = "attachment; filename=Disease_Report.txt"
    return response

if __name__ == '__main__':
    app.run(debug=True)