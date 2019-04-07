from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
import pandas as pd

import doc.nlp_otr.script as script
import doc.nlp_otr.nlp_otr as nlp


def index(request):
    return JsonResponse({'message': 'docs endpoint working'})

def test(request):
    result = script.script()
    
    return JsonResponse({'message': result})

@csrf_exempt
def upload_doc(request):
    if "GET" == request.method:
        return JsonResponse({
            'message': 'upload a file'
        })
    csv_file = request.FILES['File']

    csv = pd.read_csv(csv_file)
    
    
    # nlp_results = nlp.nlp_otr(csv_file)
    
    
    
    print(csv.info())
    print(csv.head(0))
    nlp_results = csv.head(2).to_dict()
    
    

    return JsonResponse({
        'message': nlp_results
    })