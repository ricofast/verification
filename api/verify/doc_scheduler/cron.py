from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.models import Document
from api.verify.views import preprocess_image
import keras_ocr
import pandas as pd
import os
import json
import glob
import shutil

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
    docs = Document.objects.filter(verified=True, scanned=False).values("user", "keyword", "file")
    # doc = Document.objects.create(user=19, keyword='Test 22')
    df_docs = pd.DataFrame(list(docs))
    images = []
    print(df_docs)
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

        # for i in range(len(df)-1):
        #     kw = df_docs.loc[j, "keyword"]
        #     nameexist = kw in df.loc[i, 'text']
        nameexist = kw in df['text'].values
        if nameexist:
            status["user"].append(str(userid))
            obj, created = Document.objects.update_or_create(
                user=userid,
                defaults={'scanned': True},
            )
            # with open('verifieds.json', 'a') as f:
            #     f.write(str(userid))
            #     f.write('\n')
        # else:
        #     status.append("Unverified")

        with open('verified.json', 'w') as f:
            json.dump(status, f)

        # print(status)
    # for doc in docs:
    #     userid = doc.user
    #     kw = doc.keyword
    #     df_doc1 = pd.DataFrame({"userid": [userid], "keyword": [kw]})
    #     df_doc2
    #     print("user Id:")
    #     print(userid)
    #     print(kw)
    #     path = os.getcwd() + "/media/documents/user_" + str(userid) + "/*"


        # for path_to_document in glob.glob(path, recursive=True):






def start():
    scheduler.add_job(period, "interval", minutes=20, id="unverifiedusers_001",
                    replace_existing=True)
    scheduler.start()