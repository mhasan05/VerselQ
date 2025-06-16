from django.contrib import admin
from .models import UserAuth
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('email','full_name', 'profile_pic','otp', 'otp_expired','is_verified', 'is_active','is_staff','is_superuser', 'date_joined')
    
admin.site.register(UserAuth,CustomUserAdmin)
# Register your models here.