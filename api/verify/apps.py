from django.apps import AppConfig
from api.verify.doc_scheduler import cron

class VerifyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.verify'

    def ready(self):
        print("Starting Scheduler ...")
        cron.start()
