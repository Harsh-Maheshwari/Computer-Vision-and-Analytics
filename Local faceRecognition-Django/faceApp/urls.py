from django.urls import path
from . import views
from django.http import StreamingHttpResponse
from camera import VideoCamera, gen

urlpatterns = [
	path('', views.index, name='index'),
	path('register/', views.register, name='register'),
	path('built_model/', views.build_model, name='build_model'),
	path('add_photo/',views.add_photo,name='add_photo'),
	path('capture_image/<slug:employee_id>/',views.capture_img,name='cap_img'),
    path('monitor/',views.monitor ,name='monitor' ),
    # path('monitor/', lambda r: StreamingHttpResponse(gen(VideoCamera()),content_type='multipart/x-mixed-replace; boundary=frame'),name='monitor' ),

]