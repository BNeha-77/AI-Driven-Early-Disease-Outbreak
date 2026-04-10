from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import UploadFileForm
from .ml_model.predict import predict_outbreak
from .utils.preprocessing import preprocess_dataset
import pandas as pd
import os
import joblib
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

DATA_PATH = os.path.join(settings.BASE_DIR, 'media', 'uploaded.csv')
PREPROCESSED_PATH = os.path.join(settings.BASE_DIR, 'media', 'preprocessed.csv')

def home(request):
    return render(request, 'alertapp/home.html')

@login_required
def user_home(request):
    return render(request, 'alertapp/user_home.html')

@login_required
def upload(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(request.FILES['file'])
            df.to_csv(DATA_PATH, index=False)
            messages.success(request, "Dataset uploaded successfully.")
            return redirect('user_home')
    else:
        form = UploadFileForm()
    return render(request, 'alertapp/upload.html', {'form': form})

@login_required
def preprocess(request):
    if not os.path.exists(DATA_PATH):
        messages.error(request, "Please upload a dataset first.")
        return redirect('user_home')
    df = pd.read_csv(DATA_PATH)
    df_clean = preprocess_dataset(df)
    df_clean.to_csv(PREPROCESSED_PATH, index=False)
    num_rows = df_clean.shape[0]
    return render(request, 'alertapp/preprocess.html', {'num_rows': num_rows})

@login_required
def run_algorithm(request, algo):
    if not os.path.exists(PREPROCESSED_PATH):
        messages.error(request, "Please preprocess the dataset first.")
        return redirect('user_home')
    df = pd.read_csv(PREPROCESSED_PATH)
    if 'outbreak' not in df.columns:
        messages.error(request, "The 'outbreak' column is missing in your dataset.")
        return redirect('user_home')
    # Use errors='ignore' to avoid KeyError if columns are missing
    X = df.drop(['date', 'region', 'outbreak'], axis=1, errors='ignore')
    y = df['outbreak'] 

    if algo == 'logistic':
        model = LogisticRegression()
        algo_name = "Logistic Regression"
    elif algo == 'randomforest':
        model = RandomForestClassifier()
        algo_name = "Random Forest"
    elif algo == 'svm':
        model = SVC()
        algo_name = "SVM"
    elif algo == 'knn':
        model = KNeighborsClassifier()
        algo_name = "KNN"
    else:
        messages.error(request, "Unknown algorithm.")
        return redirect('mlalgorithm')

    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()  # <-- Update here
    cr = classification_report(y, y_pred, output_dict=False)
    return render(request, 'alertapp/algorithm_result.html', {
        'algo_name': algo_name,
        'accuracy': acc,
        'confusion_matrix': cm,  # Pass as list
        'classification_report': cr,
    })

@login_required
def mlalgorithm(request):
    return render(request, 'alertapp/mlalgorithm.html')

@login_required
def mlalgorithm_analysis(request):
    if not os.path.exists(PREPROCESSED_PATH):
        messages.error(request, "Please preprocess the dataset first.")
        return redirect('user_home')
    df = pd.read_csv(PREPROCESSED_PATH)
    if 'outbreak' not in df.columns:
        messages.error(request, "The 'outbreak' column is missing in your dataset.")
        return redirect('user_home')
    X = df.drop(['date', 'region', 'outbreak'], axis=1, errors='ignore')
    y = df['outbreak']

    algos = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    results = []
    for name, model in algos.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred).tolist()
        results.append({
            'name': name,
            'accuracy': acc,
            'confusion_matrix': cm,
        })
    return render(request, 'alertapp/mlalgorithm_analysis.html', {'results': results})

@login_required
def best_algorithm(request):
    if not os.path.exists(PREPROCESSED_PATH):
        messages.error(request, "Please preprocess the dataset first.")
        return redirect('user_home')
    df = pd.read_csv(PREPROCESSED_PATH)
    if 'outbreak' not in df.columns:
        messages.error(request, "The 'outbreak' column is missing in your dataset. Please check your data.")
        return redirect('user_home')
    X = df.drop(['date', 'region', 'outbreak'], axis=1, errors='ignore')
    y = df['outbreak']
    algos = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    best_name = None
    best_acc = 0
    best_cm = None
    best_cr = None
    for name, model in algos.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_cm = confusion_matrix(y, y_pred)
            best_cr = classification_report(y, y_pred, output_dict=False)
    return render(request, 'alertapp/best_algorithm.html', {
        'algo_name': best_name,
        'accuracy': best_acc,
        'confusion_matrix': best_cm,
        'classification_report': best_cr,
    })
    
@login_required
def graphical_report(request):
    if not os.path.exists(PREPROCESSED_PATH):
        messages.error(request, "Please preprocess the dataset first.")
        return redirect('user_home')
    df = pd.read_csv(PREPROCESSED_PATH)
    if 'outbreak' not in df.columns:
        messages.error(request, "The 'outbreak' column is missing in your dataset. Please check your data.")
        return redirect('user_home')
    X = df.drop(['date', 'region', 'outbreak'], axis=1, errors='ignore')
    y = df['outbreak']
    algos = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }
    accuracies = {}
    for name, model in algos.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracies[name] = accuracy_score(y, y_pred)

    # Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(accuracies.values(), labels=accuracies.keys(), autopct='%1.1f%%')
    ax1.set_title('Algorithm Accuracy Distribution')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    pie_chart = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)

    # Line chart
    fig2, ax2 = plt.subplots()
    ax2.plot(list(accuracies.keys()), list(accuracies.values()), marker='o')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Algorithm Accuracy Comparison')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    line_chart = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)

    return render(request, 'alertapp/graphical_report.html', {
        'pie_chart': pie_chart,
        'line_chart': line_chart,
        'accuracies': accuracies,
    })
    
@login_required
def prediction(request):
    prediction_result = None
    probability = None
    risk_level = None
    display_message = None
    fever = cough = headache = search_trend = None

    if request.method == 'POST':
        fever = int(request.POST.get('fever'))
        cough = int(request.POST.get('cough'))
        headache = int(request.POST.get('headache'))
        search_trend = float(request.POST.get('search_trend'))
        features = [fever, cough, headache, search_trend]

        # Binary prediction
        prediction_result = predict_outbreak(features)

        # Probability score (if model supports predict_proba)
        try:
            from alertapp.ml_model.predict import MODEL_PATH
            import pickle
            import numpy as np
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(np.array([features]))[0][1]
                probability = round(proba * 100, 2)
            else:
                probability = None
        except Exception:
            probability = None

        # Risk level based on probability
        if probability is not None:
            if probability >= 70:
                risk_level = "High Risk"
            elif probability >= 40:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"
        else:
            risk_level = "N/A"

        # Display message
        if prediction_result == 1:
            display_message = "🚨 ALERT: Potential Outbreak Detected"
        elif prediction_result == 0:
            display_message = "✅ Status: Normal (No Outbreak)"
        else:
            display_message = "Prediction unavailable."

    return render(request, 'alertapp/predict.html', {
        'prediction': prediction_result,
        'probability': probability,
        'risk_level': risk_level,
        'display_message': display_message,
        'fever': fever,
        'cough': cough,
        'headache': headache,
        'search_trend': search_trend,
    })

from django.contrib.auth import logout

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')
    
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('user_home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'alertapp/login.html')

def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        password2 = request.POST.get('password2')
        if password != password2:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already exists.')
        else:
            User.objects.create_user(username=username, password=password)
            messages.success(request, 'Registration successful. Please log in.')
            return redirect('login')
    return render(request, 'alertapp/register.html')