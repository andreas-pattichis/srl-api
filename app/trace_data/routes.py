from fastapi import APIRouter, Response
from tortoise.expressions import Q
from tortoise import connections

from .utils import *

from app.trace_data.models import TraceData, MdlUser, MdlCourse

router = APIRouter(prefix="/api/tracedata", tags=["tracedata"])

@router.get(
    "/{username}",
    status_code=200,
)
async def tracedata_results_from_user(username: str, response: Response):
    db_moodle = connections.get('moodle')

    user = await MdlUser.get(username=username, using_db=db_moodle)
    if not user:
        response.status_code = 404
        return {
            'message': 'User not found'
        }


    course_ids = await TraceData.filter(user_id=user.id, process_label__isnull=False).distinct().values('course_id')

    results = []

    for course_id in course_ids:
        trace_data = await TraceData.filter(user_id=user.id, process_label__isnull=False, course_id=course_id['course_id']).order_by('save_time')
        trace_data = await model_to_df(trace_data)
        df = await map_process_labels(trace_data)

        # making the data series and percentages for meta and cog
        m_series, m_percentages = await create_series(df, "Metacognition")
        c_series, c_percentages = await create_series(df, "Cognition")
        series, percentages = await create_series(df, "Combined")

        course = await MdlCourse.get(id=course_id['course_id'], using_db=db_moodle)

        result = {
            'course_id': course_id['course_id'],
            'name': course.fullname if course else "Essay "+str(course_id['course_id']),
            'meta': m_series,
            'm_perc': m_percentages,
            'cog': c_series,
            'c_perc': c_percentages,
            'combined_perc': percentages,
            'combined_series': series,
        }

        results.append(result)

    return results
