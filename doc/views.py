from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

import pandas as pd

def index(request):
    return JsonResponse({'message': 'docs endpoint working'})

@csrf_exempt
def upload_doc(request):
    if "GET" == request.method:
        return JsonResponse({
            'message': 'upload a file'
        })
    csv_file = request.FILES['File']

    csv = pd.read_csv(csv_file)
    
    print(csv.info())

    
    return JsonResponse({
        'message': 'file successfully imported'
    })