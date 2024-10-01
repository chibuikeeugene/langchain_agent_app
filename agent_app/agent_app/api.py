# for observability and monitoring
from langsmith import traceable

from fastapi import FastAPI, Request
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from langserve import add_routes

# #================== app definition =======================#
# app = FastAPI(
#   title="LangChain Server",
#   version="1.0",
#   description="A simple API server using LangChain's Runnable interfaces",
# )


# #================= adding chain routes ====================#
# class Input(BaseModel):
#     input: str
#     chat_history: list[BaseMessage] = Field(
#         ...,
#         extra={"widget": {"type": "chat", "input": "location"}},
#     )

# class Output(BaseModel):
#     output: str


# add_routes(
#     app,
#     agent_executor.with_types(input_type=Input, output_type=Output),
#     path="/agent",
# )


# import uvicorn

    # uvicorn.run(app, host="localhost", port=8000)