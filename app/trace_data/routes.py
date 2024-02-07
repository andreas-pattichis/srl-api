from fastapi import APIRouter
from tortoise.contrib.fastapi import HTTPNotFoundError
from fastapi import FastAPI, HTTPException
from tortoise.expressions import Q

from .utils import *

from app.trace_data.models import TraceData

router = APIRouter(prefix="/api/tracedata", tags=["tracedata"])


@router.get(
    "/",
    response_model=list[pydantic_model()],
    responses={404: {"model": HTTPNotFoundError}}
)
async def essay_list():
    return await pydantic_model().from_queryset(TraceData.all())


@router.get(
    "/results/{study}/{username}",
    responses={404: {"model": HTTPNotFoundError}}
)
async def tracedata_results_from_user(username: str, study: str):
    username = username.replace("_", "")

    course_ids = await TraceData.filter(firstname=username, lastname=study, process_label__isnull=False).distinct().values('course_id')

    if len(course_ids) == 0:
        raise HTTPException(status_code=404, detail="User not found")

    results = []

    for course_id in course_ids:
        trace_data = await TraceData.filter(firstname=username, lastname=study, process_label__isnull=False, course_id=course_id['course_id']).order_by('save_time')
        try:  # making the pattern dataframe
            trace_data = await model_to_df(trace_data)
            df = await map_process_labels(trace_data)

        except Exception as e:
            print(e)

        # making the data series and percentages for meta and cog
        m_series, m_percentages = await create_series(df, "Metacognition")
        c_series, c_percentages = await create_series(df, "Cognition")
        series, percentages = await create_series(df, "Combined")

        result = {
            'course_id': course_id['course_id'],
            'name': "Essay "+str(course_id['course_id']),
            'meta': m_series,
            'm_perc': m_percentages,
            'cog': c_series,
            'c_perc': c_percentages,
            'combined_perc': percentages,
            'combined_series': series,
        }

        results.append(result)

    return {
        'body': results
    }
