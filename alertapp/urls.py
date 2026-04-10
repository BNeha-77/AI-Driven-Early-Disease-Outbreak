from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('preprocess/', views.preprocess, name='preprocess'),
    path('mlalgorithm/', views.mlalgorithm, name='mlalgorithm'),
    path('mlalgorithm_analysis/', views.mlalgorithm_analysis, name='mlalgorithm_analysis'),
    path('run_algorithm/<str:algo>/', views.run_algorithm, name='run_algorithm'),
    path('best_algorithm/', views.best_algorithm, name='best_algorithm'),
    path('graphical_report/', views.graphical_report, name='graphical_report'),
    path('prediction/', views.prediction, name='prediction'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('user_home/', views.user_home, name='user_home'),
    path('logout/', views.logout_view, name='logout'),
]