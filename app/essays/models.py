from tortoise.models import Model
from tortoise import fields


class Essay(Model):
    user_id = fields.CharField(max_length=255)
    save_time = fields.CharField(max_length=255)
    username = fields.CharField(max_length=255)
    url = fields.CharField(max_length=255)
    essay_content = fields.CharField(max_length=255)
    essay_content_json = fields.CharField(max_length=255)