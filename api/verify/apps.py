from django.apps import AppConfig


class VerifyConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api.verify'

    def ready(self):

        print("Starting Scheduler ...")
        from media import cron
        cron.start()
