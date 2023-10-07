from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.models import Document
from api.verify.views import preprocess_image
import keras_ocr
import pandas as pd
import os
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
    docs = Document.objects.filter(verified=True, scanned=False).values("user", "keyword")
    # doc = Document.objects.create(user=19, keyword='Test 22')
    df_docs = pd.DataFrame(list(docs))
    print(df_docs)
    path_of_docs = []
    for ind in df_docs.index:
        path = os.getcwd() + "/media/documents/user_" + str(df_docs["user"][ind]) + "/*"
        for path_to_document in glob.glob(path, recursive=True):
            path_of_docs = [path_of_docs.append(img) for img in [path_to_document]]
    pipeline = keras_ocr.pipeline.Pipeline()
    images = [keras_ocr.tools.read(img) for img in [path_of_docs]]
    prediction_groups = pipeline.recognize(images)
    df = pd.DataFrame(prediction_groups, columns=['text', 'bbox'])
    status = ""
    for i in range(len(df)):
        kw = df_docs.loc[i, "keyword"]
        nameexist = kw in df.loc[i, 'text']
        if nameexist:
            status = "Verified"
            with open('verifieds.txt', 'a') as f:
                f.writelines(df_docs.loc[i, "user"])


        print(status)
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
    scheduler.add_job(period, "interval", minutes=10, id="unverifiedusers_001",
                    replace_existing=True)
    scheduler.start()