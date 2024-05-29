from app.essays.models import SubmitAnswers, Cluster, ReflectiveResponse
from app.trace_data.models import TraceData, MdlUser, MdlCourse
from fastapi import FastAPI, HTTPException, APIRouter, Response
from tortoise.transactions import in_transaction
from tortoise import connections
from tortoise.functions import Max

router = APIRouter(prefix="/api/essays", tags=["essays"])

@router.post(
    "/submit",
    response_model=SubmitAnswers,  # We'll reflect the input for simplicity
    status_code=201,
)
async def submit_essay(data: SubmitAnswers, response: Response):
    db_moodle = connections.get('moodle')

    user = await MdlUser.get(username=data.user_id, using_db=db_moodle)
    if not user:
        response.status_code = 404
        return {
            'message': 'User not found'
        }

    for response in data.responses:
        await ReflectiveResponse.create(
            user_id=user.id,
            essay_id=data.essay_id,
            question_id=response.question_id,
            answer=response.answer
        )

    return data

@router.get(
    "/{username}",
    status_code=202,
)
async def get_essay(username: str, response: Response):
    db_moodle = connections.get('moodle')

    user = await MdlUser.get(username=username, using_db=db_moodle)
    if not user:
        response.status_code = 404
        return {
            'message': 'User not found'
        }

    course_ids = await ReflectiveResponse.filter(user_id=user.id).distinct().values('essay_id')

    results = []

    for course_id in course_ids:
        ### Retrieve only last responses of one essay if duplicate are present
        # Step 1: Find the maximum 'id' for each 'question_id'
        max_ids = await ReflectiveResponse \
            .annotate(max_id=Max('id')) \
            .filter(user_id=user.id, essay_id=course_id['essay_id']) \
            .group_by('question_id') \
            .values('question_id', 'max_id')

        # Step 2: Retrieve the specific records with these maximum 'id's
        max_id_list = [entry['max_id'] for entry in max_ids]
        essay_data = await ReflectiveResponse.filter(id__in=max_id_list).order_by('question_id')

        answers = []
        for essay in essay_data:
            answer = {"question": essay.question_id,
                      "answer": essay.answer}
            answers.append(answer)

        result = {
            'course_id': course_id['essay_id'],
            'reflective_questions': answers,
        }
        results.append(result)

    return results