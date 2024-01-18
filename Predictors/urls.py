from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name = 'index'),
    path('lung_cancer', views.lung_cancer, name = 'lung_cancer'),
    path('results', views.results, name = 'results'),
    path('pneumonia', views.pneumonia, name = 'pneumonia'),
    path('results2', views.results2, name = 'results2')
]