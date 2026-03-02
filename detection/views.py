from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import Prediction
from .ml_logic import predict_disease
import json
from django.http import JsonResponse
import google.generativeai as genai
import os

def home(request):
    diseases = [
        {
            'name': 'Tomato Late Blight',
            'image': 'images/diseases/tomato_late_blight.png',
            'symptoms': 'Large water-soaked patches on leaves; white fungal growth in humid conditions.',
            'prevention': 'Air circulation, avoiding wet foliage, immediate removal of infected plants.'
        },
        {
            'name': 'Apple Scab',
            'image': 'images/diseases/apple_scab.png',
            'symptoms': 'Circular olive-green/brown spots on leaves and fruit.',
            'prevention': 'Pruning, removing fallen leaves, fungicide application.'
        },
        {
            'name': 'Grape Black Rot',
            'image': 'images/diseases/grape_black_rot.png',
            'symptoms': 'Small brown spots on leaves; berries shrivel into hard black "mummies".',
            'prevention': 'Sanitation (removing mummies), pruning for airflow, fungicides.'
        },
        {
            'name': 'Potato Early Blight',
            'image': 'images/diseases/potato_early_blight.png',
            'symptoms': 'Dark spots with concentric rings ("bullseye") on older leaves.',
            'prevention': 'Proper spacing, drip irrigation, fungicide treatment.'
        },
        {
            'name': 'Corn Common Rust',
            'image': 'images/diseases/corn_common_rust.png',
            'symptoms': 'Small, cinnamon-brown pustules on both leaf surfaces.',
            'prevention': 'Resistant varieties, crop rotation, early planting.'
        },
        {
            'name': 'Peach Bacterial Spot',
            'image': 'images/diseases/peach_bacterial_spot.png',
            'symptoms': 'Water-soaked spots on leaves; fruit pitting and cracking.',
            'prevention': 'Resistant cultivars, balanced fertilization, copper sprays.'
        },
        {
            'name': 'Pepper Bacterial Spot',
            'image': 'images/diseases/pepper_bacterial_spot.png',
            'symptoms': 'Small, dark, water-soaked spots on leaves and fruit.',
            'prevention': 'Disease-free seeds, rotation, avoid overhead irrigation.'
        },
        {
            'name': 'Strawberry Leaf Scorch',
            'image': 'images/diseases/strawberry_leaf_scorch.jpg',
            'symptoms': 'Dark purplish spots on leaves; scorch-like appearance as they merge.',
            'prevention': 'Clean planting stock, adequate spacing, fungicide use.'
        },
        {
            'name': 'Cherry Powdery Mildew',
            'image': 'images/diseases/cherry_powdery_mildew.jpg',
            'symptoms': 'White, powdery fungal growth on leaves and shoots.',
            'prevention': 'Pruning for light and air, sulfur-based sprays.'
        },
        {
            'name': 'Citrus Greening (HLB)',
            'image': 'images/diseases/orange_haunglongbing.jpg',
            'symptoms': 'Yellowing of leaf veins (blotchy mottle); small, lopsided fruit.',
            'prevention': 'Controlling Asian Citrus Psyllid, removing infected trees, nutrient support.'
        },
    ]
    return render(request, 'detection/home.html', {'diseases': diseases})

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'registration/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')
@login_required
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']

        prediction_obj = Prediction.objects.create(
            user=request.user,
            image=image,
            result="Processing..."
        )

        # ML inference
        res, conf = predict_disease(prediction_obj.image.path)

        # ✅ FORCE STRING CONVERSION
        result_text = str(res)
        if isinstance(res, (list, tuple)):
            result_text = res[0]

        error_message = None
        if result_text == "NOT_A_PLANT":
            error_message = "Wrong image! Please upload a valid plant image."
            prediction_obj.result = "Invalid Image (Not a Plant)"
            prediction_obj.confidence = 0.0
        else:
            prediction_obj.result = result_text
            prediction_obj.confidence = float(conf)
        
        prediction_obj.save()

        context = {
            'pk': prediction_obj.pk,
            'condition_result': result_text if not error_message else None,
            'error_message': error_message,
            'confidence': f"{conf:.2f}",
            'image_url': prediction_obj.image.url
        }

        return render(request, 'detection/predict.html', context)

    return render(request, 'detection/predict.html')
@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'detection/history.html', {'predictions': predictions})

@login_required
def delete_prediction(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    prediction.delete()
    return redirect('history')

def about_view(request):
    return render(request, 'detection/about.html')

def contact_view(request):
    return render(request, 'detection/contact.html')

@login_required
def chatbot_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_message = data.get('message', '')
            
            if not user_message:
                return JsonResponse({'error': 'Empty message'}, status=400)

            api_key = os.getenv('AI_AGENT_API_KEY')
            if not api_key:
                return JsonResponse({'error': 'AI API Key not configured'}, status=500)

            genai.configure(api_key=api_key)
            
            # Dynamic model selection to avoid 404/not supported errors
            try:
                available_models = [m.name for m in genai.list_models() 
                                    if 'generateContent' in m.supported_generation_methods]
                
                # Prefer flash models
                model_name = 'models/gemini-1.5-flash' # Default fallback
                flash_models = [m for m in available_models if 'flash' in m]
                if flash_models:
                    model_name = flash_models[0]
                elif available_models:
                    model_name = available_models[0]
            except Exception:
                model_name = 'models/gemini-pro' # Legacy fallback

            model = genai.GenerativeModel(model_name)
            
            system_prompt = """
            You are a helpful and knowledgeable Plant Health Expert for 'PlantScan AI'. 
            Your goal is to help users identify plant diseases, provide care tips, and explain plant science. 
            Keep your responses concise, professional, and friendly. 
            If asked about something unrelated to plants or gardening, politely steer the conversation back to plants.
            """
            
            response = model.generate_content(f"{system_prompt}\n\nUser: {user_message}\nAssistant:")
            
            return JsonResponse({'reply': response.text})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)
