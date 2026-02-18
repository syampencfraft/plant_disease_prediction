import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

MODEL_PATH = 'plant_disease_model.keras'
MAPPING_PATH = 'class_indices.json'
API_KEY = os.getenv('AI_AGENT_API_KEY')

_model = None
_class_names = None

# Configure Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)

def load_resources():
    global _model, _class_names
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                _model = tf.keras.models.load_model(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: {MODEL_PATH} not found.")

    if _class_names is None:
        if os.path.exists(MAPPING_PATH):
            with open(MAPPING_PATH, 'r') as f:
                _class_names = json.load(f)
        else:
            print(f"Warning: {MAPPING_PATH} not found.")

def predict_disease(image_path):
    # Try AI Agent (Gemini) first
    if API_KEY:
        try:
            # Dynamic model selection to avoid 404 errors
            available_models = [m.name for m in genai.list_models() 
                                if 'generateContent' in m.supported_generation_methods]
            
            # Prefer flash models
            model_name = 'models/gemini-1.5-flash' # Default
            flash_models = [m for m in available_models if 'flash' in m]
            if flash_models:
                model_name = flash_models[0]
            elif available_models:
                model_name = available_models[0]

            model = genai.GenerativeModel(model_name)
            img = Image.open(image_path)
            
            prompt = """
            Identify the plant disease in this image. 
            Provide the name of the disease and a confidence score between 0 and 100.
            Format the response as JSON: {"disease": "Disease Name", "confidence": 95.5}
            If the plant is healthy, say "Healthy".
            If you are unsure, provide your best guess.
            """
            
            response = model.generate_content([prompt, img])
            # Extract JSON from response
            text = response.text
            try:
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != -1:
                    data = json.loads(text[start:end])
                    return data.get('disease', 'Unknown'), float(data.get('confidence', 90.0))
                else:
                    return text.strip()[:100], 90.0
            except Exception:
                return text.strip()[:100], 90.0
        except Exception as e:
            print(f"AI Agent Error: {e}")
            # Fallback to Dummy/Local ML logic below

    # --- Local ML / Dummy Logic Fallback ---
    load_resources()
    
    # Preprocess image
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        if _model is not None and _class_names is not None:
            predictions = _model.predict(img_array)
            class_idx = str(np.argmax(predictions[0]))
            confidence = float(predictions[0][int(class_idx)]) * 100
            
            # Get class name from mapping
            raw_name = _class_names.get(class_idx, "Unknown Disease")
            result = raw_name.replace('___', ' ').replace('_', ' ')
            return result, confidence
    except Exception as e:
        print(f"Local ML Fallback Error: {e}")

    return "Healthy/Unknown (AI Agent Unavailable)", 0.0
