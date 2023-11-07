from fastapi import APIRouter
from tortoise.contrib.fastapi import HTTPNotFoundError
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
    # loading the maps for colour and labels of each pattern id
    sub_dict, main_dict, color_dict = load_label_meanings()

    username = username.replace("_", "")

    print(username)
    print(study)

    course_ids = await TraceData.filter(firstname=username, lastname=study, process_label__isnull=False).distinct().values('course_id')
    print(course_ids)

    if len(course_ids) == 0:
        return {
                'statusCode': 404,
            }

    results = []

    for course_id in course_ids:
        trace_data = await TraceData.filter(firstname=username, lastname=study, process_label__isnull=False, course_id=course_id['course_id']).order_by('save_time')
        try:  # making the pattern dataframe
            trace_data = await model_to_df(trace_data)
            df, time_scaler = await load_process_features_study(sub_dict,
                                                                main_dict,
                                                                color_dict,
                                                                trace_data)

        except Exception as e:
            print(e)

        # making the data series and percentages for meta and cog
        m_series, m_perc, m_personal = await create_series(df, "Metacognition", time_scaler)
        c_series, c_perc, c_personal = await create_series(df, "Cognition", time_scaler)
        series, perc, personal = await create_series(df, "Combined", time_scaler)

        result = {
            'course_id': course_id['course_id'],
            'name': "Essay "+str(course_id['course_id']),
            'meta': m_series,
            'm_perc': m_perc,
            'cog': c_series,
            'c_perc': c_perc,
            'combined_perc': perc,
            'combined_series': series,
        }

        results.append(result)

    return {
        'statusCode': 200,
        'body': results
    }
