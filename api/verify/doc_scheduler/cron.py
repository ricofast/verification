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
import datetime
from datetime import datetime
# import glob
import shutil
from django.utils import timezone

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

    all_docs = Document.objects.filter(verified=True)
    doc = all_docs.filter((Q(name_checked=False) & Q(name__isnull=False)) | (Q(dob_checked=False) & Q(dob__isnull=False)))
    doc_name_dob = doc.values("user", "name", "dob", "file", "name_checked", "dob_checked")
    doc_name = all_docs.filter(name__isnull=False, name_checked=False).values("user", "name", "file").values("user", "name", "file")
    doc_dbo = all_docs.filter(dob__isnull=False, dob_checked=False).values("user", "dob", "file")

    df_doc = pd.DataFrame(list(doc))
    df_docs_name_dob = pd.DataFrame(list(doc_name_dob))
    df_docs_name = pd.DataFrame(list(doc_name))
    df_docs_dob = pd.DataFrame(list(doc_dbo))

    pipeline = keras_ocr.pipeline.Pipeline()
    url = "https://7yq1ahwwq0.execute-api.us-east-1.amazonaws.com/document-verified"
    header = {
        "Content-Type": "application/json",
        "Authorization": "4b338f063102cc66e604b12941bbefc2fad15840ec7ef98442028edba64ce98a",
    }
    if df_docs_name_dob.size > 0:
        df_docs = df_docs_name_dob
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

        images = [keras_ocr.tools.read(img) for img in path_of_docs]
        prediction_groups = pipeline.recognize(images)
        print("predictions Done")
        status = {"user": [],
                  "Name": [], "DOB": []}
        for j in range(len(prediction_groups)):
            df = pd.DataFrame(prediction_groups[j], columns=['text', 'bbox'])
            print("predictions:", df['text'].values)
            dobchecked = df_docs.loc[j, "dob_checked"]
            namechecked = df_docs.loc[j, "name_checked"]
            userid = df_docs.loc[j, "user"]
            print("User: ", userid)
            print("dob_checked: ", dobchecked)
            print("name_checked: ", namechecked)
            print("Name: ", df_docs.loc[j, "name"])

            if namechecked is False:

                kw = df_docs.loc[j, "name"]
                print("name: ", kw)
                name_status = 2
                if kw:
                    # check keyword with multiple words
                    words = kw.split()
                    allkeywords_status = False

                    for wd in words:

                        print("keyword:", wd)

                        nameexist = wd in df['text'].values
                        print("Exist: ", nameexist)
                        if nameexist:
                            allkeywords_status = True
                            name_status = 1
                        else:
                            lwd = wd.lower()
                            similar = difflib.get_close_matches(lwd, df['text'].values)
                            print(similar)
                            if len(similar) > 0:
                                allkeywords_status = True
                                name_status = 1
                            else:
                                allkeywords_status = False
                                name_status = 2
                                break
            else:
                name_status = 1
            if dobchecked is False:
                kw = df_docs.loc[j, "dob"]
                dob_status = 2
                if kw:
                    # check keyword with multiple words
                    words = kw.split()
                    allkeywords_status = False

                    for wd in words:
                        print("keyword:", wd)

                        nameexist = wd in df['text'].values
                        print("Exist: ", nameexist)
                        if nameexist:
                            allkeywords_status = True
                            dob_status = 1
                        else:
                            lwd = wd.lower()
                            similar = difflib.get_close_matches(lwd, df['text'].values)
                            print(similar)
                            if len(similar) > 0:
                                allkeywords_status = True
                                dob_status = 1
                            else:
                                allkeywords_status = False
                                dob_status = 2
                                break
            else:
                dob_status = 1


            if dob_status == 1 and name_status == 1:
                payload = {
                    "user": userid,
                    "result": 1
                }

                result = requests.post(url, data=json.dumps(payload), headers=header)

                if result.status_code == 200:
                    status["user"].append(str(userid))
                    status["Name"].append("Yes")
                    status["DOB"].append("Yes")
                    obj, created = Document.objects.update_or_create(
                        user=userid, defaults={'name_checked': True, 'dob_checked': True, 'name_received': True, 'dob_received': True}, )

            elif dob_status == 1 and name_status == 2:
                payload = {
                    "user": userid,
                    "result": 2
                }

                result = requests.post(url, data=json.dumps(payload), headers=header)
                if result.status_code == 200:
                    status["user"].append(str(userid))
                    status["Name"].append("No")
                    status["DOB"].append("Yes")
            elif dob_status == 2 and name_status == 1:
                payload = {
                    "user": userid,
                    "result": 3
                }

                result = requests.post(url, data=json.dumps(payload), headers=header)
                if result.status_code == 200:
                    status["user"].append(str(userid))
                    status["Name"].append("Yes")
                    status["DOB"].append("No")


            else:
                payload = {
                    "user": userid,
                    "result": 4
                }

                result = requests.post(url, data=json.dumps(payload), headers=header)
                if result.status_code == 200:
                    status["user"].append(str(userid))
                    status["Name"].append("No")
                    status["DOB"].append("No")
        print("status: ", status)
        dt = datetime.datetime.now()
        seq = int(dt.strftime("%Y%m%d%H%M%S"))
        filename = f'media/verified/verified-athlete-{seq}.json'
        with open(filename, 'w') as f:
            json.dump(status, f)

    # if df_docs_dob.size > 0:
    #     df_docs = df_docs_dob
    #     images = []
    #     print(df_docs)
    #
    #     path_of_docs = []
    #     for ind in df_docs.index:
    #         # path = os.getcwd() + "/media/documents/user_" + str(df_docs["user"][ind]) + "/*"
    #         path = os.getcwd() + "/media/" + df_docs["file"][ind]
    #         path_of_docs.append(path)
    #         # for path_to_document in glob.glob(path, recursive=True):
    #         #     path_of_docs = path_of_docs.append(path_to_document)
    #     print(path_of_docs)
    #     # print(os.getcwd())
    #
    #     images = [keras_ocr.tools.read(img) for img in path_of_docs]
    #     prediction_groups = pipeline.recognize(images)
    #     status = {"user": [],
    #               "passed": []}
    #     for j in range(len(prediction_groups)):
    #         df = pd.DataFrame(prediction_groups[j], columns=['text', 'bbox'])
    #         userid = df_docs.loc[j, "user"]
    #         kw = df_docs.loc[j, "dob"]
    #         if kw:
    #             # check keyword with multiple words
    #             words = kw.split()
    #             allkeywords_status = False
    #             for wd in words:
    #                 nameexist = wd in df['text'].values
    #                 if nameexist:
    #                     allkeywords_status = True
    #                 else:
    #                     lwd = wd.lower()
    #                     similar = difflib.get_close_matches(lwd, df['text'].values)
    #                     if len(similar) > 0:
    #                         allkeywords_status = True
    #                     else:
    #                         allkeywords_status = False
    #                         break
    #             # for i in range(len(df)-1):
    #             #     kw = df_docs.loc[j, "keyword"]
    #             #     nameexist = kw in df.loc[i, 'text']
    #             # nameexist = kw in df['text'].values
    #             if allkeywords_status:
    #
    #                 payload = {
    #                     "user": userid,
    #                     "result": "3"
    #                 }
    #
    #                 result = requests.post(url, data=json.dumps(payload), headers=header)
    #
    #                 if result.status_code == 200:
    #                     status["user"].append(str(userid))
    #                     status["passed"].append("Yes")
    #                     obj, created = Document.objects.update_or_create(
    #                         user=userid, defaults={'dob_checked': True, 'dob_received': True}, )
    #             else:
    #                 payload = {
    #                     "user": userid,
    #                     "result": "4"
    #                 }
    #                 result = requests.post(url, data=json.dumps(payload), headers=header)
    #                 if result.status_code == 200:
    #                     status["user"].append(str(userid))
    #                     status["passed"].append("No")
    #                 # with open('verifieds.json', 'a') as f:
    #                 #     f.write(str(userid))
    #                 #     f.write('\n')
    #             # else:
    #             #     status.append("Unverified")
    #     dt = datetime.datetime.now()
    #     seq = int(dt.strftime("%Y%m%d%H%M%S"))
    #     filename = f'media/verified/verified-dob-{seq}.json'
    #     with open(filename, 'w') as f:
    #         json.dump(status, f)


def deletedoc(userid):

    folder = os.getcwd() + '/media/documents/user_' + str(userid) + '/'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))




def delete_unverified():
    all_docs = Document.objects.filter(verified=False)

    for doc in all_docs:
        if (timezone.now()-doc.uploaded).days >= 1:
            deletedoc(doc.user)
            doc.delete()


def start():
    scheduler.add_job(period, 'cron', hour=23, minute=59, id="unverifiedusers_001", replace_existing=True)
    scheduler.add_job(delete_unverified, 'interval', minutes=2, id="unverifiedusers_002", replace_existing=True)
    scheduler.start()