from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
import pandas as pd



from io import StringIO

import doc.nlp_otr.script as script
import doc.nlp_otr.nlp_otr as nlp
import doc.nlp_otr.main as main


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
    data_string = str(request.body.decode('utf-8'))
    print('****************************************************************************')
    
    print(data_string)
    print(type(data_string))

    data = StringIO(data_string)
    
    # print(data)
    # print(type(data))
    
    csv_file = pd.read_csv(data, sep=',')

    print(type(csv_file))
    # csv_file = request.FILES['File']
    
    # csv = pd.read_csv(csv_file)
    
    # print(csv_file)
    
    # nlp_results = csv.head(2).to_dict()
    

    return JsonResponse({
        'message': ''
    })

def nlptest(request):
    result = main.jsonfunc()
    return JsonResponse({'result':result})