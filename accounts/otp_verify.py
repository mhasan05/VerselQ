
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from email.utils import formataddr
import random


def send_otp(email,otp):
    """
    Function to send OTP to the user's email.
    """
    # Email configuration
    try:
        subject = 'Verify Your Identity'
        from_email = formataddr(("Morpheous", settings.EMAIL_HOST_USER))
        to = email
        html_content = render_to_string('otp_verification.html',{'otp':otp})
        text_content = strip_tags(html_content)
        msg = EmailMultiAlternatives(subject,text_content, from_email, [to])
        msg.attach_alternative(html_content, "text/html")
        msg.send()
        return True
    except:
        return False
    