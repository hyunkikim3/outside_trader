from django.contrib import admin


from . models import Data, Market, KNN, PLS, Logistic, RandomForest, Bagging, Boosting, PCR, Tree, Transaction, Result

# Register your models here.
admin.site.register(Data)
admin.site.register(Market)
admin.site.register(KNN)
admin.site.register(PLS)
admin.site.register(Logistic)
admin.site.register(RandomForest)
admin.site.register(Bagging)
admin.site.register(Boosting)
admin.site.register(PCR)
admin.site.register(Tree)
admin.site.register(Transaction)
admin.site.register(Result)