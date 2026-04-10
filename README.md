# AI-Driven Early Disease Outbreak Alert System

This Django-based project uses syndromic surveillance and web search trends to predict and alert about early disease outbreaks using supervised machine learning.

## Features
- User and admin authentication
- Dataset upload, preprocessing, and ML analysis
- Multiple supervised ML algorithms with metrics and graphical reports
- Admin user management

## Setup
1. Install requirements: `pip install -r requirements.txt`
2. Set up `.env` with your secrets.
3. Run migrations: `python manage.py migrate`
4. Create a superuser: `python manage.py createsuperuser`
5. Start the server: `python manage.py runserver`