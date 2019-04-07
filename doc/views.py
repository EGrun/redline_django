from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import pickle



from io import StringIO

import doc.nlp_otr.script as script
import doc.nlp_otr.nlp_otr as nlp
import doc.nlp_otr.main as main


def index(request):
    return JsonResponse({'message': 'docs endpoint working'})

@csrf_exempt
def upload_doc(request):
    if "GET" == request.method:
        return JsonResponse({
            'message': 'upload a file'
        })
    
    
    try:
        data_string = str(request.body.decode('utf-8'))

        data = StringIO(data_string)
        
        csv_file = pd.read_csv(data, sep=',')

    except:
        print('failed to decode string')
    
    try:
        file = request.FILES['File']

        csv_file = pd.read_csv(file)
    except:
        print('failed to decode file')
        return JsonResponse({'message': 'failed to decode file'})


    dbfile = open('pandas_df', 'ab')
    pickle.dump(csv_file, dbfile)
    dbfile.close()
    
    

    return JsonResponse({
        'data': csv_file.head(2).to_dict()
    })

def nlptest(request):
    result = main.jsonfunc()
    return JsonResponse({'result':result})