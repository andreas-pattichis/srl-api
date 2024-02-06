from tortoise.models import Model
from tortoise import fields


class TraceData(Model):
    id = fields.IntField(pk=True)
    user_id = fields.IntField()
    course_id = fields.IntField()
    save_time = fields.BigIntField()
    process_label = fields.CharField(max_length=255)

    class Meta:
        table = "trace_data"

    def __str__(self):
        return self.username

class MdlUser(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=255)

    class Meta:
        table = "mdl_user"

    def __str__(self):
        return self.username
    
class MdlCourse(Model):
    id = fields.IntField(pk=True)
    fullname = fields.CharField(max_length=255)
    shortname = fields.CharField(max_length=255)

    class Meta:
        table = "mdl_course"

    def __str__(self):
        return self.fullname