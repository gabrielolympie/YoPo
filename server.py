import json
import time

from typing import Optional, List, AsyncGenerator
from pydantic import BaseModel

from starlette.responses import StreamingResponse
from fastapi import FastAPI, APIRouter
from dotenv import load_dotenv
import os
import uuid
from core.streaming_pipeline_old import StreamingPipeline

load_dotenv('.env')

app = FastAPI(title="OpenAI-compatible API")

os.environ['OPENAI_API_KEY'] = os.environ['DEFAULT_API_KEY']

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "mock-gpt-model"
    pipeline_name: str = "message_dict"
    messages: List = []
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

async def _resp_async_generator(text_generator: AsyncGenerator[str, None], request: ChatCompletionRequest):
    async for token in text_generator:
        chunk = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "created": time.time(),
            "model": request.model,
            "choices": [{"delta": {"content": token}}],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "role": "assistant",
                        "content": token,
                        "tool_calls": None,
                        "tool_calls_json": None
                    },
                    "logprobs": None
                }
            ],
            "usage": None
        }
        yield f"data: {json.dumps(chunk)}\n\n"

# Create a router with the prefix /v1
v1_router = APIRouter()
@v1_router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    pipeline = StreamingPipeline(pipeline_path=os.path.join('pipelines', request.pipeline_name + '.yaml'))
    
    input_data = {
        "message_processor": {"messages": request.messages},
    }
    
    print(input_data)
    response_generator = pipeline.process_user_input(input_data)
    if request.stream:
        return StreamingResponse(
            _resp_async_generator(response_generator, request=request), media_type="application/x-ndjson"
        )

    else:
        return {
            "id": "1337",
            "object": "chat.completion",
            "created": time.time(),
            "model": request.model,
            "choices": [{"message": Message(role="assistant", content=response_generator)}],
        }

# Include the router in the main app
app.include_router(v1_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)