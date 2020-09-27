from django.http import StreamingHttpResponse
from camera import VideoCamera, gen
from django.shortcuts import render, redirect
from camera import gen_emb, cap_img, VideoCamera
from .forms import RegForm
from .models import regData,detectedData
import os

camera1 = VideoCamera()

def index(request):
	return render(request, 'faceApp/index.html')

def register(request):
	if request.method == 'POST':
		reg_form = RegForm(data=request.POST)
		if reg_form.is_valid():
			reg_form.save()
	else:
		reg_form = RegForm()

	return render(request,'faceApp/register.html',{'reg_form': reg_form})

def add_photo(request):
	employee_id = regData.objects.latest('reg_date').employee_id
	return render(request, 'faceApp/add_photo.html', {'employee_id':employee_id})

def capture_img(request,employee_id):
	global camera1
	directory = os.path.join('./dataset/',str(employee_id))

	isdir = os.path.isdir(directory)
	if not isdir:
		os.mkdir(directory)

	count = cap_img(camera1,directory)
	if count >= 3:
		# gen_emb()
		return redirect('/')
	if count < 3:
		all_var = []
		return render(request, 'faceApp/add_photo.html', {'employee_id':employee_id})

def build_model(request):
	gen_emb()
	return render(request, 'faceApp/index.html')


def monitor(request):
	global camera1
	return StreamingHttpResponse(gen(camera1),content_type='multipart/x-mixed-replace; boundary=frame')



