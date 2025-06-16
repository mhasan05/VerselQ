from rest_framework import serializers
from .models import UserAuth


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserAuth
        fields = ['id', 'email', 'full_name','profile_pic','otp','is_verified','is_active','is_staff','is_superuser','date_joined']
