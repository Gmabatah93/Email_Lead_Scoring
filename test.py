"""
Path Parameters:
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}

Query Parameters: 
- When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters.
- The query is the set of key-value pairs that go after the ? in a URL, separated by & characters.
@app.get("/items/")
async def read_item(q: str | None = None):
    if q:
        return {"q": q}
    return {"q": "No query string provided"}

Request Body:
"""
from typing import Annotated
from fastapi import FastAPI, Query

app = FastAPI()


@app.get("/items/")
async def read_items(q: Annotated[str | None, Query(max_length=50)] = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results