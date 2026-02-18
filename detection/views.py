from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from .models import Prediction
from .ml_logic import predict_disease

def home(request):
    return render(request, 'detection/home.html')

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

        prediction_obj.result = result_text
        prediction_obj.confidence = float(conf)
        prediction_obj.save()

        context = {
            'condition_result': result_text,
            'confidence': f"{conf:.2f}",
            'image_url': prediction_obj.image.url
        }

        return render(request, 'detection/predict.html', context)

    return render(request, 'detection/predict.html')
