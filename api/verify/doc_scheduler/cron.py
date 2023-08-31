from apscheduler.schedulers.background import BackgroundScheduler
from api.verify.models import Document

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
    docs = Document.objects.all()
    print(docs)

def start():
    scheduler.add_job(period, "interval", minutes=1, id="unverifiedusers_001",
                    replace_existing=True)
    scheduler.start()