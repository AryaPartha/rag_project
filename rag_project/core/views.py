from django.shortcuts import render
from django.http import HttpResponse
from .models import Document

def upload_view(request):

    if request.method == "POST":
        print("FILES:", request.FILES)  # DEBUG LINE

        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return HttpResponse("No file received!")

        doc = Document(file=uploaded_file)
        doc.save()

        return HttpResponse("File uploaded successfully!")

    return render(request, "core/upload.html")
