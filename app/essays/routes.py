from fastapi import APIRouter
from tortoise.contrib.fastapi import HTTPNotFoundError
from .utils import *

from app.essays.models import Essay

router = APIRouter(prefix="/essays", tags=["essays"])


@router.get(
    "/",
    response_model=list[pydantic_model()],
    responses={404: {"model": HTTPNotFoundError}}
)
async def essay_list():
    return await pydantic_model().from_queryset(Essay.all())


@router.get(
    "/{username}",
    response_model=pydantic_model(),
    responses={404: {"model": HTTPNotFoundError}}
)
async def essay_from_user(username: str):
    return await pydantic_model().from_queryset_single(Essay.get(username=username))


@router.get(
    "/results/{username}",
    responses={404: {"model": HTTPNotFoundError}}
)
async def essay_results_from_user(username: str):
    essay = await essay_from_user(username)

    # loading the maps for colour and labels of each pattern id
    sub_dict, main_dict, color_dict = load_label_meanings()

    try:  # making the pattern dataframe
        df, time_scaler = load_process_features_study_f(sub_dict,
                                                        main_dict,
                                                        color_dict,
                                                        username + "_pattern.csv")
    except Exception as e:
        print(e)
        return {
            'statusCode': 400,
        }
        pass

    return essay
