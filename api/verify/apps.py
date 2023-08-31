from django.apps import AppConfig
from .cron import start

class VerifyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.verify'

    def ready(self):
        print("Starting Scheduler ...")
        start()
