import hmac
import hashlib
import glob
import os.path as osp
from base64 import b64encode
from dateutil import parser as date_parser
import torch
import cv2
import dlib
import numpy as np
from django.conf import settings
from super_gradients.training import models
from . import RRDBNet_arch as arch
import torchvision.transforms as transforms
import PIL.Image as Image
from PIL import ImageStat
from mtcnn import MTCNN
from numba import cuda
import gc


static_folder = settings.STATIC_ROOT
media_folder = settings.MEDIA_ROOT

picture_enhance_model = "models/RRDB_ESRGAN_x4.pth"


def extract_text_with_pyPDF(PDF_File):

    pdf_reader = PdfReader(PDF_File)

    raw_text = ''

    for i, page in enumerate(pdf_reader.pages):

        text = page.extract_text()
        if text:
            raw_text += text

    return raw_text


def is_date_parsing(date_str):
  try:
    return bool(date_parser.parse(date_str))
  except ValueError:
    return False


def verifySignature(receivedSignature: str, secret, params):

  data = "-".join(params)
  data = data.encode('utf-8')
  computed_sig = hmac.new(secret.encode('utf-8'), msg=data, digestmod=hashlib.sha256).digest()
  signature = b64encode(computed_sig).decode()
  if signature == receivedSignature:
    return True
  return False


def set_device():
  if torch.cuda.is_available():
    dev = "cuda:0"
  else:
    dev = "cpu"
  return torch.device(dev)


def classify(aimodel, image_transforms, grayimage_transforms, image_path, classes):
  aimodel = aimodel.eval()
  image = Image.open(image_path)
  im = image.convert("RGB")
  stat = ImageStat.Stat(im)
  # if sum(stat.sum) / 3 != stat.sum[0]:
  image = image_transforms(im).float()
  image = image.unsqueeze(0)
  # else:
  #   image = grayimage_transforms(im).float()
  #   image = image.unsqueeze(0)

  output = aimodel(image)
  _, predicted = torch.max(output.data, 1)
  # predicted : "Birth Certificate" or "ID / DL" or "Invalide"
  return classes[predicted.item()]


def is_head_shot_clear(image_path, threshold=20):
  # path = os.getcwd() + "/media/headshots/user_" + str(userid) + "/*"
  # image = ""
  # for image_path in glob.glob(path, recursive=True):
    # Load the image using OpenCV
  image = cv2.imread(image_path)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Calculate the Variance of Laplacian to measure image clarity
  variance_of_laplacian = cv2.Laplacian(gray_image, cv2.CV_64F).var()

  # Determine if the image is clear based on the threshold
  is_clear = variance_of_laplacian > threshold
  # is_clear = variance_of_laplacian

  return is_clear

def dlib_headfacerecognize(image):
  detector = dlib.get_frontal_face_detector()
  img = dlib.load_rgb_image(image)
  rects = detector(img, 1)
  if rects == 1:
    return True
  else:
    return False


def headshots_count(image_path):
  # path = os.getcwd() + "/media/headshots/user_" + str(userid) + "/*"
  # image = ""
  # for image_path in glob.glob(path, recursive=True):
    # Load the image using OpenCV
  #image = cv2.imread(image_path)

  # Convert the image to grayscale
  image = cv2.imread(image_path)
  image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Method 1
  # Call Yolo V4 to detect objects in the image
  # print("Start image detection")
  # boxes, label, count = cv.detect_common_objects(image)
  # print("Number of boxes")
  # print(len(boxes))
  ## output = draw_bbox(image, box, label, count)

# Method 2
  # Call Yolo V8 to detect objects in the image
  # sts.update({'runs_dir': yolov8_run})
  # sts.reset()
  # model = YOLO(yolov8_model + "yolov8s.pt")
  # results = model.predict(source=image_path, conf=0.3)
  # boxes = results[0].boxes

# Method 3
  # call Yolo Nas to detect objects in the image
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  MODEL_ARCH = 'yolo_nas_l'
  model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)
  CONFIDENCE_TRESHOLD = 0.10
  result = list(model.predict(image1, conf=CONFIDENCE_TRESHOLD))[0]
  dp = result.prediction
  boxes = dp.bboxes_xyxy
  # Determine if the image is clear based on the threshold
  one_person = len(boxes) == 1
  class_id = dp.labels.astype(int)
  count = np.count_nonzero(class_id == 0)
  verified = 0
  if (one_person and class_id[0] == 0):
    if dlib_headfacerecognize(image_path):
    # nfc = check_face(image1)
    # if nfc > 0:
      verified = 1


  # del model
  # gc.collect()
  # torch.cuda.empty_cache()
  #
  # device = cuda.get_current_device()
  # device.reset()



  # elif one_person == 1 and not class_id[0] == 0:
  #   verified = 2
  # elif count == 1:
  #   verified = 1
  # else:
  #   verified = 0
  # gc.collect()
  # with torch.no_grad():
  # print("classe: ", class_id)
  # print("Type: ", type(class_id))
  # print("count: ", count)
  # print("verified :", verified)
  # print("one_persone :", one_person)
  #   torch.cuda.empty_cache()
  return verified


def is_image_clear(image_path, threshold=3.5):
  # Load the image using Pillow (PIL)
  image = Image.open(image_path).convert('RGB')

  # Convert the image to a PyTorch tensor
  image_tensor = torch.tensor([transforms.ToTensor()(image)])

  # Load the pre-trained NIQE model
  niqe_model = niqe()

  # Calculate the NIQE score for the image
  niqe_score = niqe_model(image_tensor).item()

  # Determine if the image is clear based on the threshold
  is_clear = niqe_score < threshold

  return is_clear


def enhancepictures(userid):
  model_path = picture_enhance_model  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
  device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
  # device = torch.device('cpu')

  print("current folder")
  print(model_path)
  test_img_folder = media_folder + "/documents/user_" + str(userid) + "/*"
  test_user_folder = media_folder + "/documents/user_" + str(userid) + "/"
  print(test_user_folder)
  model = arch.RRDBNet(3, 3, 64, 23, gc=32)
  model.load_state_dict(torch.load(model_path), strict=True)
  model.eval()
  model = model.to(device)

  print('Model path {:s}. \nTesting...'.format(model_path))

  idx = 0
  im_path = ""
  for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    print("Finished")
    with torch.no_grad():
      output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    print("Write picture to file")
    cv2.imwrite('{pth}{file}_rlt.png'.format(pth=test_user_folder, file=base), output)
    im_path = '{pth}{file}_rlt.png'.format(pth=test_user_folder, file=base)
  return im_path


def check_face(image):
  detector = MTCNN()
  faces = detector.detect_faces(image)
  num_faces = len(faces)

  return num_faces