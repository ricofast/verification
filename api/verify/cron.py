from api.verify.models import Document, HeadShot


def document_ocr():
    # docs = Document.objects.filter(verified=True, scanned=True)
    # with open('verifiedids.txt', 'a') as f:
    #     for doc in docs:
    #         f.writelines(doc.pk)
    doc = Document.objects.create(user=9, keyword='Test 1')
    doc.save()
    return
