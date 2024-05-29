from pydantic import BaseModel
from typing import List
from tortoise import Tortoise, fields
from tortoise.models import Model


# Define Pydantic models for incoming data

class ClusterIn(BaseModel):
    cluster_id: int
    probability: float


class ResponseIn(BaseModel):
    question_id: int
    answer: str

class SubmitAnswers(BaseModel):
    user_id: str
    essay_id: int
    responses: List[ResponseIn]

class SubmitCluster(BaseModel):
    user_id: str
    essay_id: int
    clusters: List[ClusterIn]


class SubmitEssayRequest(BaseModel):
    user_id: str
    essay_id: int
    clusters: List[ClusterIn]
    responses: List[ResponseIn]


# Tortoise ORM models
class ReflectiveResponse(Model):
    id = fields.IntField(pk=True)
    user_id = fields.IntField(index=True)
    # user_id = fields.CharField(max_length=255, index=True)
    essay_id = fields.IntField(index=True)
    question_id = fields.IntField(index=True)
    answer = fields.TextField()

    class Meta:
        table = "responses_data"


class Cluster(Model):
    id = fields.IntField(pk=True)
    user_id = fields.CharField(max_length=255, index=True)
    essay_id = fields.IntField(index=True)
    cluster_id = fields.IntField(index=True)
    probability = fields.FloatField()

    class Meta:
        table = "clusters_data"
