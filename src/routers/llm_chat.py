from fastapi import APIRouter, Response, Query, status, Depends, Request
from typing import Annotated, Dict, Any, Optional
from pydantic import BaseModel

from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory
from src.handlers.api_key_auth_handler import APIKeyAuth
from src.utils.config import settings
from src.schemas.response import BasicResponse, ChatResponse
from fastapi.responses import StreamingResponse
from collections.abc import AsyncGenerator
import json

# API key authentication instance
api_key_auth = APIKeyAuth()
router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    question_input: str
    model_name: str = 'deepseek-r1'
    collection_name: str = settings.QDRANT_COLLECTION_NAME
    use_multi_collection: bool = False


@router.post("/chat", response_description="Chat with LLM system", response_model=ChatResponse)
async def chat_with_llm(
    request: Request,
    response: Response,
    chat_request: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Process chat requests with organization information
    resp = await ChatHandler().handle_request_chat(
        session_id=chat_request.session_id,
        question_input=chat_request.question_input,
        model_name=chat_request.model_name,
        collection_name=chat_request.collection_name,
        user_id=user_id,
        organization_id=organization_id,
        use_multi_collection=chat_request.use_multi_collection
    )
                                           
    if resp.status == "Success" and resp.data:
        response.status_code = status.HTTP_200_OK
        content = resp.data if isinstance(resp.data, str) else str(resp.data)
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=content
        )
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ChatResponse(
            id=chat_request.session_id,
            role="assistant",
            content=f"Error: {resp.message}"
        )


async def format_sse(generator) -> AsyncGenerator[str, None]:
    """
    Format async generator thành chuẩn Server-Sent Events
    """
    async for chunk in generator:
        if chunk:
            # Format according to SSE standard
            yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"

@router.post("/chat/stream/sse", response_description="Chat with LLM system (SSE format)")
async def chat_with_llm_stream_sse(
    request: Request,
    chat_request: ChatRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    return StreamingResponse(
        format_sse(
            ChatHandler().handle_streaming_chat(
                session_id=chat_request.session_id,
                question_input=chat_request.question_input,
                model_name=chat_request.model_name,
                collection_name=chat_request.collection_name,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=chat_request.use_multi_collection
            )
        ),
        media_type="text/event-stream"
    )

@router.post("/{user_id}/create_session", response_description="Create session")
async def create_session(
    request: Request,
    response: Response,
    user_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    organization_id = getattr(request.state, "organization_id", None)
    
    request_user_id = getattr(request.state, "user_id", None)
    if request_user_id != user_id:
        user_role = getattr(request.state, "role", None)
        if user_role != "ADMIN":
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="Failed",
                message="You can only create sessions for yourself",
                data=None
            )
    
    resp = ChatHandler().create_session_id(
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.data:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.post("/{session_id}/delete_history", response_description="Delete history of session id")
async def delete_chat_history(
    request: Request,
    response: Response,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        
    Returns:
        JSON response indicating success or failure
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    resp = ChatMessageHistory().delete_message_history(
        session_id=session_id,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.post("/{session_id}/get_chat_history", response_description="Chat history of session id")
async def chat_history_by_session_id(
    request: Request,
    response: Response,
    session_id: str,
    limit: int = 10,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        limit: Maximum number of messages to retrieve (default: 10)
        
    Returns:
        JSON response with the chat history
    """
    # Get user_id and organization_id information from request state
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    # Call the get_list_message_history method with the appropriate parameters
    resp = ChatMessageHistory().get_list_message_history(
        session_id=session_id,
        limit=limit,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp