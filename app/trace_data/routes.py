from fastapi import APIRouter, Response
# from tortoise.expressions import Q
# from tortoise import connections
import requests

from .utils import *
# from app.trace_data.models import TraceData, MdlUser, MdlCourse

router = APIRouter(prefix="/api/tracedata", tags=["tracedata"])

@router.get(
    "/{username}",
    status_code=200,
)
async def tracedata_results_from_user(username: str, response: Response):
    url = f'https://nijmegen.floraproject.org/api/tracedata/{username}'
    # Send the GET request
    response_api = requests.get(url)
    # Check if the request was successful
    if response_api.status_code == 200:
        # Get JSON data from the response
        data = response_api.json()
        # print(data)
    else:
        # print('User not found')
        response.status_code = 404
        return {
            'message': 'User not found'
        }

    results = []

    for course in data:
        cluster_names, cluster_probs = clustering_results(course)
        # Add these to the json_data dictionary
        course['cluster_names'] = cluster_names
        course['cluster_probs'] = cluster_probs
        results.append(course)

    # print(results)

    return results
