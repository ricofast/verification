from django.shortcuts import render
import os, shutil
import glob
import cv2
import re
import pytesseract
from django.http import JsonResponse
# Create your views here.
from rest_framework import authentication, permissions
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser, FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer, FileScanSerializer
from django.views.decorators.csrf import csrf_exempt
from .models import Document, AIModel
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image


classes = [
          'Birth Certificate',
          'ID/DL',
          'Invalid']

filename = rf"media/aimodels/best_model.pth"
ai_model = torch.load(filename)

mean = [0.7683, 0.7671, 0.7645]
std = [0.2018, 0.1925, 0.1875]

image_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


def classify(aimodel, image_transforms, image_path, classes):
  aimodel = aimodel.eval()
  image = Image.open(image_path)
  image = image_transforms(image).float()
  image = image.unsqueeze(0)

  output = aimodel(image)
  _, predicted = torch.max(output.data, 1)

  return classes[predicted.item()]



class FileUpdateView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  # parser_classes = (FileUploadParser,)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)

    if file_serializer.is_valid():
      userid = file_serializer.data['user']

      doc = Document.objects.filter(user=userid).first()
      if doc:
        delete(userid)

      key_word = file_serializer.data['keyword']
      filename = file_serializer.validated_data['file']
      print("file: ",  filename)
      obj, created = Document.objects.update_or_create(
        user=userid,
        defaults={'keyword': key_word, 'file': filename},
      )
      # file_serializer.save()

      document = obj
      # verified = Scanpicture(document.keyword, document.user)

      verified = classify(ai_model, image_transforms, document.file, classes)
      if verified != "invalid":
        verified = verified + "--" + Scanpicture(document.keyword, document.user)

      return Response(verified, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# Create your views here.


class DocumentScanView(APIView):
  parser_classes = (MultiPartParser, FormParser)

  @csrf_exempt
  def post(self, request, *args, **kwargs):
    file_serializer = FileScanSerializer(data=request.data)

    if file_serializer.is_valid():
      userid = file_serializer.data['user']

      key_word = file_serializer.data['keyword']
      print(key_word)
      print(userid)
      verified = Scanpicture(key_word, userid)

      return Response(verified, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)



def find_string(text, target_string):
  # Convert both the text and target string to lowercase for case-insensitive matching
  text_lower = text.lower()
  target_string_lower = target_string.lower()

  # Search for the target string within the text
  if target_string_lower in text_lower:
    return True
  else:
    return False


def Scanpicture(athname, userid):
  # athname = request.POST.get('athname')
  # path = os.getcwd() + "/media/images/*"
  path = os.getcwd() + "/media/images/user_" + str(userid) + "/*"
  print(path)
  filter_predicted_result = ""
  for path_to_document in glob.glob(path, recursive=True):
    # img = cv2.imread(path_to_document)
    print(path_to_document)
    img = preprocess_image(path_to_document)
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    predicted_result = pytesseract.image_to_string(img, lang='eng')
    # predicted_result = pytesseract.image_to_string(img, lang='eng',config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    filter_predicted_result = "".join(predicted_result.split("\n")).replace(":", "").replace("-", "")

  words = athname.split()
  # status = {}
  #
  # for wd in words:
  #   nameexist = find_string(filter_predicted_result, wd)
  #   if nameexist:
  #     status[wd] = "Verified"
  #   else:
  #     status[wd] = "Unverified"

  status = ""

  for wd in words:
    nameexist = find_string(filter_predicted_result, wd)
    if nameexist:
      status = status + wd + " Verified - "
    else:
      status = status + wd+ " Unverified - "

  # context = {'filter_predicted_result': filter_predicted_result, 'name': name}

  return status
  # context = {'form': form}
  # return render(request, 'homepage.html', context)


def delete(userid):
  folder = os.getcwd() + '/media/images/user_' + str(userid) + '/'
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))


def preprocess_image(image_path):
  # Load the image
  image = cv2.imread(image_path)

  # Resize the image for better OCR accuracy
  resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

  # Denoise the image using a bilateral filter
  denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

  # Apply adaptive thresholding to enhance contrast
  threshold_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)

  return threshold_image


