from django.http import HttpRequest, HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def home(req: HttpRequest) -> HttpResponse:
    return render(req, "stock/home.html")