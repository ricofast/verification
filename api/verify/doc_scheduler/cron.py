from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.models import Document, KerasModelLoaded
#  from api.verify.views import preprocess_image
from django.db.models import Q
import keras_ocr
import pandas as pd
import os
import json
import requests
import difflib
# import glob
# import shutil

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
    # keras_loaded = KerasModelLoaded.objects.first()
    # if keras_loaded and keras_loaded.loaded == False:
    #     pipeline = keras_ocr.pipeline.Pipeline()
    #     keras_loaded.loaded = True
    #     keras_loaded.save()
    # elif not keras_loaded:
    #     pipeline = keras_ocr.pipeline.Pipeline()
    #     aicreated = KerasModelLoaded.objects.create(loaded=True)


    doc = Document.objects.filter(Q(verified=True) & ((Q(name__isnull=False) & Q(name_checked=False)) | (Q(dob__isnull=False) & Q(dob_checked=False))))
    docs = doc.values("user", "keyword", "file")

    df_docs = pd.DataFrame(list(docs))
    images = []
    print(df_docs)
    url = "https://7yq1ahwwq0.execute-api.us-east-1.amazonaws.com/document-verified"
    header = {
        "Content-Type": "application/json",
        "Authorization": "4b338f063102cc66e604b12941bbefc2fad15840ec7ef98442028edba64ce98a",
    }
    path_of_docs = []
    for ind in df_docs.index:
        # path = os.getcwd() + "/media/documents/user_" + str(df_docs["user"][ind]) + "/*"
        path = os.getcwd() + "/media/" + df_docs["file"][ind]
        path_of_docs.append(path)
        # for path_to_document in glob.glob(path, recursive=True):
        #     path_of_docs = path_of_docs.append(path_to_document)
    print(path_of_docs)
    # print(os.getcwd())
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(img) for img in path_of_docs]
    prediction_groups = pipeline.recognize(images)
    status = {"user":[]}
    for j in range(len(prediction_groups)):
        df = pd.DataFrame(prediction_groups[j], columns=['text', 'bbox'])
        userid = df_docs.loc[j, "user"]
        kw = df_docs.loc[j, "keyword"]
        if kw:
            # check keyword with multiple words
            words = kw.split()
            allkeywords_status = False
            for wd in words:
                nameexist = wd in df['text'].values
                if nameexist:
                    allkeywords_status = True
                else:
                    lwd = wd.lower()
                    similar = difflib.get_close_matches(lwd, df['text'].values)
                    if len(similar) > 0:
                        allkeywords_status = True
                    else:
                        allkeywords_status = False
                        break
            # for i in range(len(df)-1):
            #     kw = df_docs.loc[j, "keyword"]
            #     nameexist = kw in df.loc[i, 'text']
            # nameexist = kw in df['text'].values
            if allkeywords_status:

                payload = {
                    "user": userid,
                    "result": "1"
                }

                result = requests.post(url, data=json.dumps(payload), headers=header)

                if result.status_code == 200:
                    status["user"].append(str(userid))

                    obj, created = Document.objects.update_or_create(user=userid, defaults={'scanned': True},)

                # with open('verifieds.json', 'a') as f:
                #     f.write(str(userid))
                #     f.write('\n')
            # else:
            #     status.append("Unverified")

            with open('media/verified/verified.json', 'w') as f:
                json.dump(status, f)



def start():
    scheduler.add_job(period, "interval", hours=2, id="unverifiedusers_001",
                    replace_existing=True)
    scheduler.start()