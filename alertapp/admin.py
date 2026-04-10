from django.contrib import admin
from .models import UploadedData

@admin.register(UploadedData)
class UploadedDataAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_at')
    search_fields = ('file',)