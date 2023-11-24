from tortoise.models import Model
from tortoise import fields
from pathlib import Path


def query():
    f = open('./app/essays/insert.sql', mode='r', encoding='utf-8-sig')
    q = f.read()
    f.close()

    return q


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
