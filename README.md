# 🌱 SmartFarm: Intelligent Agriculture Assistant

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

**SmartFarm** is a web-based agricultural support system designed to help farmers maximize crop yield and reduce losses. It uses Machine Learning to provide accurate recommendations based on soil conditions and detects plant diseases from leaf images.

## 🚀 Live Demo
Check out the live website here: **[https://smartfarm-fhtt.onrender.com]**

---

## 🌟 Key Features

### 1. 🌾 Crop Recommendation
* **Input:** Nitrogen (N), Phosphorous (P), Potassium (K), Temperature, Humidity, pH, Rainfall.
* **Model:** Random Forest Classifier (99% Accuracy).
* **Output:** Suggests the best crop to grow in those specific conditions.

### 2. 🧪 Fertilizer Recommendation
* Analyzes soil nutrient levels and recommends the specific fertilizer needed to boost fertility.

### 3. 🍃 Plant Disease Detection
* **Input:** Upload an image of a plant leaf (Potato, Tomato, Corn, etc.).
* **Model:** Deep Learning (CNN/ResNet) trained on thousands of plant images.
* **Output:** Identifies the disease and suggests treatment.

### 4. 🌐 Multilingual Support
* Integrated **Google Translate** to support local languages (Hindi, Telugu, Tamil, etc.), making it accessible to farmers across regions.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, CSS3, Bootstrap, JavaScript.
* **Backend:** Python, Flask.
* **Machine Learning:** Scikit-learn (Random Forest), PyTorch (Disease Detection), NumPy, Pandas.
* **Deployment:** Render Cloud Hosting.

---

## 📸 Screenshots

| Crop Selection | Fertilizer Advice |
|:---:|:---:|
| <img width="940" height="482" alt="image" src="https://github.com/user-attachments/assets/a9c638e8-f58a-4162-b9ac-d5f72afe3e5c" />| <img width="940" height="484" alt="image" src="https://github.com/user-attachments/assets/821e24a3-8ba7-4c26-a93c-701fabca605a" />|

| Disease Detection | Result Page |
|:---:|:---:|
| <img width="940" height="500" alt="image" src="https://github.com/user-attachments/assets/bea37abd-b383-4f91-99dc-0ad65b87c321" />| <img width="940" height="540" alt="image" src="https://github.com/user-attachments/assets/c52b1551-58de-4ed9-9c4f-fa9d9a93e9c9" />|



---

## 💻 How to Run Locally

Run these commands to set up the project locally:

```bash
# 1. Clone the repo
git clone https://github.com/sharathbandis/SmartFarm.git
cd SmartFarm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py


📂 Project Structure
SmartFarm/
├── app.py                  # Main Flask application
├── requirements.txt        # List of dependencies
├── Procfile                # For deployment (Render/Heroku)
├── RandomForest.pkl        # Trained Crop Recommendation Model
├── Plant_Disease_Model.pth # Trained Disease Detection Model
├── fertilizer.csv          # Data for fertilizer recommendation
├── static/                 # CSS, Images, JS files
├── templates/              # HTML files
└── utils/                  # Helper scripts for ML logic


🔮 Future Scope
Mobile App: Converting the web app into a React Native mobile application.

Real-time Sensors: Integrating IoT sensors to fetch soil data automatically.

Market Prices: Adding a feature to show real-time market prices of crops.

👨‍💻 Author
Bandi Sharath

GitHub: github.com/sharathbandis
