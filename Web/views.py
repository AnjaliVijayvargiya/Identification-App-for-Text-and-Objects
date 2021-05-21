from django.shortcuts import render
from .models import Image
from .forms import ImageForm
from subprocess import run, PIPE

import sys

# Create your views here.
def index(request):
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid:
            form.save()
            img = Image.objects.last()
            #print(img.image)
            image_1 = 'media/'+ str(img.image)
            print(image_1)
            p = str(img.image)
            print(str(img.image))
            if(img.identify == "text"):
                out = run([sys.executable, 'media/python/easy_ocr.py',image_1,p], shell=True, stdout=PIPE)
                #print(out)
            else:
                out = run([sys.executable, 'media/python/obj_detect.py',image_1,p], shell=True, stdout=PIPE)
            with open('media/new.txt') as f:
                lines = f.read()
            return render(request, "home.html",{"img":img,"form":form,"p":p,"lines":lines})
    else:
        form = ImageForm()
    return render(request, 'home.html',{"form":form})