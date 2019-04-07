from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name = 'index'),
    path('upload', views.upload_doc, name='upload'),
    path('test', views.test, name = 'test'),
    path('nlptest', views.nlptest, name='nlptest')
]