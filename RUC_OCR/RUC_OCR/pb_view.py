#coding:utf-8
from django.http import HttpResponse,StreamingHttpResponse, JsonResponse
from django.shortcuts import render_to_response, render
import os,sys,time

def upload_data(request):
    state = 'no'
    global num_pro
    
    if request.POST:
        handle()
        state = 'yes'
    return render_to_response('pbtest.html', {'a':state})

def handle():
    global num_pro
    for i in range(10):
            time.sleep(2)
            num_pro = (i+1)*10
            print num_pro

def show_progress(request):
    print num_pro
    return JsonResponse(num_pro, safe=False)    
