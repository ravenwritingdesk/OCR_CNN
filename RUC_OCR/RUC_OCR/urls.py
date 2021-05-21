"""RUC_OCR URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.conf import settings
from . import view
from . import view_tar_pic
from . import pb_view
from . import OCR_unit

urlpatterns = [
    #url(r'^admin/', admin.site.urls),
    #url(r'^rendarenocr$', view.OCRweb),
    url(r'^rendarenocr$', view_tar_pic.OCRweb),
    url(r'^pbtest$', pb_view.upload_data),
    url(r'^getpb$', pb_view.show_progress),
    url(r'^get_frcnn_pro$', view_tar_pic.get_frcnn_pro),
    url(r'^get_OCR_pro$', view_tar_pic.get_OCR_pro),
    url(r'^get_select_pro$', view_tar_pic.get_select_pro),
]+static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
