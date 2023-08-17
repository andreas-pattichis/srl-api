from tortoise.models import Model
from tortoise import fields


class User(Model):
    username = fields.CharField(max_length=255)

    def __str__(self):
        return self.username