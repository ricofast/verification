from .models import Document, HeadShot


def document_ocr():
    docs = Document.objects.filter(verified=True, scanned=True)
    verified_ids = ""
    for doc in docs:
        verified_ids = doc.pk
    print(verified_ids)
    return
