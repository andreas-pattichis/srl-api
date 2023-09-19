from tortoise.models import Model
from tortoise import fields


class TraceData(Model):
    id = fields.IntField(pk=True)
    user_id = fields.IntField(null=False)
    save_time = fields.BigIntField()
    username = fields.CharField(max_length=64)
    url = fields.CharField(max_length=255)
    firstname = fields.CharField(max_length=64)
    lastname = fields.CharField(max_length=64)
    source = fields.CharField(max_length=255)
    page_event = fields.CharField(max_length=255)
    target_object = fields.CharField(max_length=255)
    instant_event = fields.CharField(max_length=255)
    sub_action_label = fields.CharField(max_length=255)
    screen_x = fields.CharField(max_length=255)
    screen_y = fields.CharField(max_length=255)
    client_x = fields.CharField(max_length=255)
    client_y = fields.CharField(max_length=255)
    window_inner_width = fields.CharField(max_length=10)
    window_inner_height = fields.CharField(max_length=10)
    screen_width = fields.CharField(max_length=10)
    screen_height = fields.CharField(max_length=10)
    event_value = fields.CharField(max_length=255)
    process_label = fields.CharField(max_length=64)
    course_id = fields.IntField()

    def __str__(self):
        return self.username
