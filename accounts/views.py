from django.contrib.auth import authenticate
from django.utils import timezone
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .models import UserAuth
from .serializers import UserSerializer
from django.conf import settings
from rest_framework.permissions import AllowAny
import uuid
from datetime import datetime
from .otp_verify import send_otp






@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request):
    if request.method == 'POST':
        email = request.data.get('email')
        password = request.data.get('password')
        full_name = request.data.get('full_name')
        if not email or not password or not full_name:
            return Response({"message": "All fields (Email, password, name) are required."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            user = UserAuth()
            user.email = email
            user.full_name = full_name
            user.set_password(password)
            user.save()
        except Exception as e:
            return Response({"error": "The email is already taken. Please provide an unique email."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate OTP
        otp = user.generate_otp()
        # Send email
        send_email = send_otp(email, otp)
        if send_email:
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            return Response({
                # 'refresh': str(refresh),
                'access': str(access_token),
                'message': 'Please verify your email using the OTP sent to your email address.'
            }, status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Failed to send OTP email. Please try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def admin_login(request):
    if request.method == 'POST':
        email = request.data.get('email')
        password = request.data.get('password')
        if not email or not password:
            return Response({'status': 'error',"message": "Both email and password are required."}, status=status.HTTP_400_BAD_REQUEST)
        user = authenticate(email=email, password=password)
        if user is not None and not user.is_verified:
            return Response({'status': 'error',"message": "Please verify your email address."}, status=status.HTTP_400_BAD_REQUEST)
        if user is not None and not user.is_superuser:
            return Response({'status': 'error',"message": "You are not authorized to access this."}, status=status.HTTP_403_FORBIDDEN)
        if user is not None:
            user_info = UserAuth.objects.get(email=email)
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            return Response({
                'status': 'success',
                # 'refresh': str(refresh),
                'access': str(access_token),
                'user_profile': UserSerializer(user_info).data
            }, status=status.HTTP_200_OK)
        else:
            return Response({'status': 'error',"message": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
def login(request):
    if request.method == 'POST':
        email = request.data.get('email')
        password = request.data.get('password')
        if not email or not password:
            return Response({'status': 'error',"message": "Both email and password are required."}, status=status.HTTP_400_BAD_REQUEST)
        user = authenticate(email=email, password=password)
        if user is not None and not user.is_verified:
            return Response({'status': 'error',"message": "Please verify your email address."}, status=status.HTTP_400_BAD_REQUEST)
        if user is not None:
            user_info = UserAuth.objects.get(email=email)
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token
            return Response({
                'status': 'success',
                # 'refresh': str(refresh),
                'access': str(access_token),
                'user_profile': UserSerializer(user_info).data
            }, status=status.HTTP_200_OK)
        else:
            return Response({'status': 'error',"message": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        


@api_view(['POST'])
def verify_email(request):
    if request.method == 'POST':
        email = request.data.get('email')
        otp = request.data.get('otp')

        if not email or not otp:
            return Response({"error": "Both email and OTP are required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = UserAuth.objects.get(email=email)
        except UserAuth.DoesNotExist:
            return Response({"error": "User does not exist."}, status=status.HTTP_404_NOT_FOUND)
        
        if timezone.now() > user.otp_expired:
            return Response({"error": "OTP has expired."}, status=status.HTTP_400_BAD_REQUEST)
        
        elif user.otp == otp:
            user.is_verified = True
            user.save()
            user = UserAuth.objects.get(email=email)
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            return Response({'status': 'success','access':access_token,"message": "Email verified successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({'status': 'error',"message": "Invalid OTP."}, status=status.HTTP_400_BAD_REQUEST)
        

@api_view(['POST'])
def resend_otp(request):
    email = request.data.get('email')
    try:
        user = UserAuth.objects.get(email=email)
    except UserAuth.DoesNotExist:
        return Response({"error": "User does not exist."}, status=status.HTTP_404_NOT_FOUND)
    otp = user.generate_otp()  
    # Send email
    send_email = send_otp(email, otp)
    if send_email:
        return Response({'status': 'success',"message": "We sent you an OTP to your email."}, status=status.HTTP_200_OK)
    else:
        return Response({'status': 'error',"message": "Failed to send OTP email. Please try again later."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    email = request.user
    user_profile = UserAuth.objects.get(email=email)

    if request.method == 'GET':
        serializer = UserSerializer(user_profile)
        return Response(serializer.data, status=status.HTTP_200_OK)



@api_view(['PATCH'])
@permission_classes([IsAuthenticated])
def update_user(request):
    data = request.data
    email = request.user.email
    try:
        user_profile = UserAuth.objects.get(email=email)
    except UserAuth.DoesNotExist:
        return Response({'status': 'error', "message": "User not found."}, status=status.HTTP_404_NOT_FOUND)

    restricted_fields = [
        'is_verified', 'email', 'is_active', 'is_staff', 
        'is_superuser', 'otp', 'otp_expired', 'date_joined'
    ]
    
    if any(field in data for field in restricted_fields) and not user_profile.is_superuser:
        return Response({
            'status': 'error',
            "message": "You cannot update these fields."
        }, status=status.HTTP_400_BAD_REQUEST)

    serializer = UserSerializer(user_profile, data=data, partial=True)

    if serializer.is_valid():
        serializer.save()
        return Response({
            'status': 'success',
            "message": "Successfully Updated Profile"
        }, status=status.HTTP_200_OK)
    else:
        return Response({
            'status': 'error',
            "message": "Invalid data provided"
        }, status=status.HTTP_400_BAD_REQUEST)





@api_view(['GET'])
@permission_classes([IsAuthenticated])
def all_user_list(request):
    user = request.user

    if user.is_superuser:
        all_user_list = UserAuth.objects.all()
        serializer = UserSerializer(all_user_list, many=True)
        return Response({
                'status': 'success',
                'total_user': len(all_user_list),
                'user_list': serializer.data
            }, status=status.HTTP_200_OK)
    else:
        return Response(
            {'status': 'error',"message": "Permission denied. Only admin can access this resource."},
            status=status.HTTP_403_FORBIDDEN
        )
    



@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    email = request.user
    old_password = request.data.get('old_password')
    new_password = request.data.get('new_password')
    confirm_password = request.data.get('confirm_password')
    user = authenticate(email=email, password=old_password)
    
    if old_password is None or new_password is None or confirm_password is None:
        return Response({'status': 'error',"message": "Please provide valid password"}, status=status.HTTP_400_BAD_REQUEST)
    
    elif new_password != confirm_password:
        return Response({'status': 'error',"message": "Password do not match."}, status=status.HTTP_400_BAD_REQUEST)

    elif user is not None:
        user.set_password(new_password)
        user.save()
    else:
        return Response({'status': 'error',"message": "Invalid old Password"}, status=status.HTTP_400_BAD_REQUEST)


    return Response({'status': 'success',"message": "Successfully change your password."}, status=status.HTTP_200_OK)




@api_view(['POST'])
@permission_classes([IsAuthenticated])
def forgot_password(request):
    new_password = request.data.get('new_password')
    confirm_password = request.data.get('confirm_password')

    if new_password != confirm_password:
        return Response({"error": "password do not match."}, status=status.HTTP_400_BAD_REQUEST)

    elif new_password is None or confirm_password is None:
        return Response({"message": "all fields are required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = request.user
        user.set_password(new_password)
        user.save()
    except:
        return Response({"message": "Invalid Email"}, status=status.HTTP_400_BAD_REQUEST)


    return Response({"message": "Successfully reset your password."}, status=status.HTTP_200_OK)