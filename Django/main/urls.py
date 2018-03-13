from django.conf.urls import url
from . import views


urlpatterns = [
	url(r'^$', views.index, name='index'),
	url(r'^choose/$', views.choose, name='choose'),
	url(r'^best/$', views.best, name='best'),
	url(r'^best_table/$', views.best_table, name='best_table'),
	url(r'^performance/$', views.performance, name='performance'),	
	url(r'^graph/$', views.graph, name='graph'),
	url(r'^result/$', views.result, name='result'),
]
