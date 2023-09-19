from fastapi import APIRouter
from tortoise.contrib.fastapi import HTTPNotFoundError

from .utils import *

from app.trace_data.models import TraceData

router = APIRouter(prefix="/tracedata", tags=["tracedata"])


@router.get(
    "/",
    response_model=list[pydantic_model()],
    responses={404: {"model": HTTPNotFoundError}}
)
async def essay_list():
    return await pydantic_model().from_queryset(TraceData.all())


@router.get(
    "/results/{username}",
    responses={404: {"model": HTTPNotFoundError}}
)
async def tracedata_results_from_user(username: str):
    # loading the maps for colour and labels of each pattern id
    sub_dict, main_dict, color_dict = load_label_meanings()

    trace_data = await TraceData.filter(username=username)

    try:  # making the pattern dataframe
        trace_data = await model_to_df(trace_data)

        df, time_scaler = await load_process_features_study(sub_dict,
                                                            main_dict,
                                                            color_dict,
                                                            trace_data)

        print("Load process features study")
        print(df.to_dict())

    except Exception as e:
        print(e)
        return {
            'statusCode': 400,
        }
        pass

    # making the data series and percentages for meta and cog
    m_series, m_perc, m_personal = await create_series(df, "Metacognition", time_scaler)
    c_series, c_perc, c_personal = await create_series(df, "Cognition", time_scaler)

    result = {
        'meta': m_series,
        'm_perc': m_perc,
        'cog': c_series,
        'c_perc': c_perc,
    }

    return {
        'statusCode': 200,
        'body': result
    }
