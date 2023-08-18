from tortoise.models import Model
from tortoise import fields


class Essay(Model):
    id = fields.IntField(pk=True)
    user_id = fields.IntField(null=False)
    save_time = fields.BigIntField()
    username = fields.CharField(max_length=255)
    url = fields.CharField(max_length=255)
    essay_content = fields.TextField()
    essay_content_json = fields.TextField()

    def __str__(self):
        return self.username
