from tortoise.models import Model
from tortoise import fields


class TraceData(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=64)
    course_id = fields.IntField()
    firstname = fields.CharField(max_length=64)
    lastname = fields.CharField(max_length=64)
    save_time = fields.BigIntField()
    process_label = fields.CharField(max_length=64)

    class Meta:
        table = "trace_data"

    def __str__(self):
        return self.username
