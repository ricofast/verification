# from api.verify.models import Document, HeadShot
from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.views import UnverifiedViewset

# def document_ocr():
#     # docs = Document.objects.filter(verified=True, scanned=True)
#     # with open('verifiedids.txt', 'a') as f:
#     #     for doc in docs:
#     #         f.writelines(doc.pk)
#     doc = Document.objects.create(user=9, keyword='Test 1')
#     doc.save()
#     return


def start():
  scheduler = BackgroundScheduler()
  unverify_docs = UnverifiedViewset()
  scheduler.add_job(unverify_docs._get_unverified_users(), "interval", minutes=1, id="unverifiedusers_001",
                    replace_existing=True)
  scheduler.start()