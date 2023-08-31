from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.models import Document
from api.verify.views import preprocess_image
import keras_ocr
import pandas as pd
import os
import glob

# def document_ocr():
#     # docs = Document.objects.filter(verified=True, scanned=True)
#     # with open('verifiedids.txt', 'a') as f:
#     #     for doc in docs:
#     #         f.writelines(doc.pk)
#     doc = Document.objects.create(user=9, keyword='Test 1')
#     doc.save()
#     return

scheduler = BackgroundScheduler()

def period():
    docs = Document.objects.filter(verified=True, scanned=True).first()
    # doc = Document.objects.create(user=19, keyword='Test 22')
    userid = docs.user
    kw = docs.keyword
    print("user Id:")
    print(userid)
    print(kw)
    path = os.getcwd() + "/media/documents/user_" + str(userid) + "/*"
    status = ""
    pipeline = keras_ocr.pipeline.Pipeline()
    for path_to_document in glob.glob(path, recursive=True):
        images = [keras_ocr.tools.read(img) for img in [path_to_document]]
        prediction_groups = pipeline.recognize(images)
        df = pd.DataFrame(prediction_groups[0], columns=['text', 'bbox'])
        nameexist = kw in df['text'].values
        if nameexist:
            status = "Verified"
            with open('verifiedids.txt', 'a') as f:
                f.writelines(docs.user)
    print(status)



def start():
    scheduler.add_job(period, "interval", minutes=10, id="unverifiedusers_001",
                    replace_existing=True)
    scheduler.start()