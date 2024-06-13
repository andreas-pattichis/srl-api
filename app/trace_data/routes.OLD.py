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
        trace_data = await TraceData.filter(user_id=user.id, process_label__isnull=False,
                                            course_id=course_id['course_id']).order_by('save_time')
        trace_data = await model_to_df(trace_data)
        # columns: process_start_time,process_end_time,process_label,process_time_spend
        df = await map_process_labels(trace_data)
        # columns: process_start_time,process_end_time,process_label,process_time_spend, process_sub, process_main, color

        # Copy df to new variable to avoid changing the original df
        df_copy = df.copy()
        # print("Save data")
        df_copy['Username'] = username
        # df_copy.to_csv('filename_username.csv', index=False)
        cluster_names, cluster_probs = clustering_results(df_copy)

        # making the data series and percentages for meta and cog
        m_series, m_percentages = await create_series(df, "Metacognition")
        c_series, c_percentages = await create_series(df, "Cognition")
        series, percentages = await create_series(df, "Combined")

        course = await MdlCourse.get(id=course_id['course_id'], using_db=db_moodle)
        name_nl = "Essay " + str(course_id['course_id'])
        name_en = "Essay " + str(course_id['course_id'])
        if course:
            if '{mlang}' in course.fullname:
                name_en = course.fullname.split('{mlang en}')[1].split('{mlang}')[0]
                name_nl = course.fullname.split('{mlang nl}')[1].split('{mlang}')[
                    0] if '{mlang nl}' in course.fullname else name_en
            else:
                name_nl = course.fullname
                name_en = course.fullname

        result = {
            'course_id': course_id['course_id'],
            'name_nl': name_nl,
            'name_en': name_en,
            'meta': m_series,
            'm_perc': m_percentages,
            'cog': c_series,
            'c_perc': c_percentages,
            'combined_perc': percentages,
            'combined_series': series,
            'cluster_names': cluster_names,
            'cluster_probs': cluster_probs
        }

        results.append(result)

    return results
