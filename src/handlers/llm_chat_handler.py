from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.retrieval_handler import SearchRetrieval
from src.helpers.llm_helper import LLMGenerator
from src.helpers.prompt_template_helper import ContextualizeQuestionHistoryTemplate, QuestionAnswerTemplate
from src.schemas.response import BasicResponse
from src.helpers.chat_management_helper import ChatService

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from typing import Optional, Dict, List, Any, Tuple
from src.database.models.schemas import ChatSessions
from src.handlers.multi_collection_retriever import multi_collection_retriever
from collections.abc import AsyncGenerator
import datetime 
import time
from typing import List, Tuple, Optional
from src.database.models.schemas import ChatSessions, Documents, ReferenceDocs

# Initialize the chat service
chat_service = ChatService()

class ChatHandler(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.search_retrieval = SearchRetrieval()
        self.llm_generator = LLMGenerator()
    
    def create_session_id(self, user_id: str, organization_id: Optional[str] = None) -> BasicResponse:
        try:
            session_id = chat_service.create_chat_session(
                user_id=user_id,
                organization_id=organization_id
            )
            self.logger.info(f"Created new chat session with ID: {session_id}")
            
            return BasicResponse(
                status="Success",
                message="Session created successfully",
                data=session_id
            )
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to create session: {str(e)}",
                data=None
            )

    async def _get_chat_flow(self, model_name: str, collection_name: str, user_id: str = None, organization_id: str = None, use_multi_collection: bool = False) -> Tuple[Runnable, Runnable]:
        """
        Create the chat flow for retrieving context and generating responses
        
        Args:
            model_name: The name of the LLM model to use
            collection_name: The name of the vector collection to query
            user_id: User ID for multi-collection access (optional)
            organization_id: Organization ID for multi-collection access (optional)
            use_multi_collection: Whether to use both personal and organizational collections
            
        Returns:
            Tuple[Runnable, Runnable]: The conversation chain and rewrite chain
        """
        # Get the language model
        llm = await self.llm_generator.get_llm(model=model_name)
        
        # Chain for rewriting the question based on conversation history
        rewrite_prompt = ContextualizeQuestionHistoryTemplate
        rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config(run_name='rewrite_chain')

        # Define the retrieval function
        async def retriever_function(query):
            if use_multi_collection and user_id:
                # Sử dụng multi-collection retriever nếu được yêu cầu
                return await multi_collection_retriever.retrieve_from_collections(
                    query=query, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                # Sử dụng retriever thông thường
                return await self.search_retrieval.qdrant_retrieval(
                    query=query, 
                    collection_name=collection_name
                )
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Main conversation chain that combines the rewritten query, context, and generates a response
        chain = (
            {
                "context": itemgetter("rewrite_input") | RunnableLambda(retriever_function).with_config(run_name='stage_retrieval') | format_docs,
                "input": itemgetter("input")
            }
            | QuestionAnswerTemplate
            | llm
            | StrOutputParser()
        ).with_config(run_name='conversational_rag')

        return chain, rewrite_chain

    # Adjust handle_request_chat method to support multi-collection
    async def handle_request_chat(
        self,
        session_id: str,
        question_input: str,
        model_name: str,
        collection_name: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False
    ) -> BasicResponse:
        """
        Handle a chat request: retrieve context, generate a response
        
        Args:
            session_id: The chat session ID
            question_input: The user's question
            model_name: The LLM model to use
            collection_name: The vector collection to query
            user_id: The user ID for multi-collection access
            organization_id: The organization ID for multi-collection access
            use_multi_collection: Whether to use both personal and organizational collections
            
        Returns:
            BasicResponse: The response to the chat request
        """
        try:
            # Get the chains needed for the chat flow
            conversational_rag_chain, rewrite_chain = await self._get_chat_flow(
                model_name=model_name, 
                collection_name=collection_name,
                user_id=user_id,
                organization_id=organization_id,
                use_multi_collection=use_multi_collection
            )

            # Save the user's question to the database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Get chat history and rewrite the question for better context
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            rewrite_input = await rewrite_chain.ainvoke(
                input={"input": question_input, "chat_history": chat_history}
            )
            
            # Retrieve context documents
            context_docs = []
            if use_multi_collection and user_id:
                context_docs = await multi_collection_retriever.retrieve_from_collections(
                    query=rewrite_input, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=rewrite_input, 
                    collection_name=collection_name
                )
            
            # Format context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
            
            # Generate the response with the context
            # Format messages using template
            messages = QuestionAnswerTemplate.format_messages(
                context=context,
                input=question_input
            )
            
            # Get LLM
            llm = await self.llm_generator.get_llm(model=model_name)
            
            # Generate response
            llm_response = await llm.ainvoke(messages)
            resp = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Calculate the response time
            response_time = round(time.time() - start_time, 3)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=resp,
                response_time=response_time
            )
            
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
            self.logger.info(f"Successfully handled chat request in session {session_id}")
            
            return BasicResponse(
                status='Success',
                message="Chat request processed successfully",
                data=resp
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle chat request: {str(e)}")
            return BasicResponse(
                status='Failed',
                message=f"Failed to handle chat request: {str(e)}",
                data=None
            )
        
    async def handle_streaming_chat(
            self,
            session_id: str,
            question_input: str,
            model_name: str,
            collection_name: str,
            user_id: str = None,
            organization_id: str = None,
            use_multi_collection: bool = False
        ) -> AsyncGenerator[str, None]:
        """
        Process chat request and return responses in streaming format
        
        Args:
            session_id: Chat session ID
            question_input: User's question
            model_name: LLM model to use
            collection_name: Vector collection name
            user_id: User ID for access control
            organization_id: Organization ID for access control
            use_multi_collection: Whether to use multiple collections
            
        Yields:
            Response chunks as they're generated
        """
        try:
            # Save user question to database
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input
            )
            
            # Create placeholder for assistant response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing response
            start_time = time.time()
            
            # Get chat history
            chat_history = ChatMessageHistory.string_message_chat_history(session_id)
            
            # Get LLM for context question rewriting
            llm = await self.llm_generator.get_llm(model=model_name)
            rewrite_chain = ContextualizeQuestionHistoryTemplate | llm | StrOutputParser()
            
            # Rewrite question for better context
            rewrite_input = await rewrite_chain.ainvoke(
                {"input": question_input, "chat_history": chat_history}
            )
            
            # Retrieve relevant context from vector database
            context_docs = []
            if use_multi_collection and user_id:
                context_docs = await multi_collection_retriever.retrieve_from_collections(
                    query=rewrite_input, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                context_docs = await self.search_retrieval.qdrant_retrieval(
                    query=rewrite_input, 
                    collection_name=collection_name
                )
            
            # Format context from retrieved documents
            context = "\n\n".join(doc.page_content for doc in context_docs) if context_docs else ""
            
            # Format messages using template
            messages = QuestionAnswerTemplate.format_messages(
                context=context,
                input=question_input
            )
            
            # Get streaming-enabled LLM
            streaming_llm = await self.llm_generator.get_llm(model=model_name)
            
            # Store complete response for database update
            full_response = []
            
            # Stream the response chunks
            async for chunk in streaming_llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    # Clean thinking sections
                    cleaned_chunk = self.llm_generator.clean_thinking(chunk.content)
                    if cleaned_chunk:
                        full_response.append(cleaned_chunk)
                        yield cleaned_chunk
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Update complete response in database
            complete_response = "".join(full_response)
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=complete_response,
                response_time=response_time
            )
            
            # Save document references safely
            if context_docs:
                await self._save_document_references(message_id, context_docs)
            
        except Exception as e:
            self.logger.error(f"Streaming chat error: {str(e)}")
            yield f"An error occurred: {str(e)}"
    
    async def _save_document_references(self, message_id: str, context_docs: List) -> None:
        """
        Safely save document references, checking for duplicates and existence in database
        
        Args:
            message_id: Message ID to associate references with
            context_docs: Context documents from vector search
        """
        try:
            # Track document IDs that have been processed for this message
            processed_doc_ids = set()
            
            # Process each document, ensuring we only process each once
            for doc in context_docs:
                if 'document_id' in doc.metadata:
                    document_id = doc.metadata['document_id']
                    
                    # Skip if already processed for this message
                    if document_id in processed_doc_ids:
                        continue
                    
                    # Mark as processed
                    processed_doc_ids.add(document_id)
                    
                    # Get page number from metadata
                    page = doc.metadata.get('index', 0)
                    
                    # Try to save, ChatService.save_reference_docs will handle the checks
                    result = chat_service.save_reference_docs(
                        message_id=message_id,
                        document_id=document_id,
                        page=page
                    )
                    
                    # Use result to avoid logging for each failure
                    if result is None:
                        # Reference wasn't saved (already exists or document not found)
                        pass
                    
        except Exception as e:
            self.logger.error(f"Error saving document references: {str(e)}")



class ChatMessageHistory(LoggerMixin):
    """
    Utility class for working with chat message history
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def messages_from_items(items: list) -> List[BaseMessage]:
        """
        Convert raw message items to BaseMessage objects
        
        Args:
            items: List of (content, type) tuples
            
        Returns:
            List[BaseMessage]: List of message objects
        """
        def _message_from_item(message: tuple) -> BaseMessage:
            _type = message[1]
            if _type == "human" or _type == "user":
                return HumanMessage(content=message[0])
            elif _type == "ai" or _type == "assistant":
                return AIMessage(content=message[0])
            elif _type == "system":
                return SystemMessage(content=message[0])
            else:
                raise ValueError(f"Got unexpected message type: {_type}")

        messages = [_message_from_item(msg) for msg in items]
        return messages

    @staticmethod
    def concat_message(messages: List[BaseMessage]) -> str:
        """
        Concatenate messages into a single string
        
        Args:
            messages: List of BaseMessage objects
            
        Returns:
            str: Concatenated message history
        """
        concat_chat = ""
        for mes in messages:
            if isinstance(mes, HumanMessage):
                concat_chat += " - user: " + mes.content + "\n"
            else:
                concat_chat += " - assistant: " + mes.content + "\n"
        return concat_chat
    
    @staticmethod
    def string_message_chat_history(session_id: str) -> str:
        """
        Get the chat history as a string
        
        Args:
            session_id: The ID of the chat session
            
        Returns:
            str: The chat history as a string
        """
        items = chat_service.get_chat_history(session_id=session_id, limit=6)
        messages = ChatMessageHistory.messages_from_items(items)
        
        # Reverse the order and skip the current message being processed
        history_str = ChatMessageHistory.concat_message(messages[::-1][:-2])
        return history_str

    def get_list_message_history(
        self, 
        session_id: str, 
        limit: int = 10, 
        user_id: Optional[str] = None, 
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Get the list of messages in the chat history
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response with message history as data
        """
        try:
            # Kiểm tra quyền truy cập nếu cung cấp user_id
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Kiểm tra xem user_id có khớp với session owner không
                    if session_info.get("user_id") != user_id:
                        # Kiểm tra xem user có thuộc organization sở hữu session không
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Người dùng thuộc tổ chức, cho phép truy cập
                            pass
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to view this chat history",
                                data=None
                            )
            
            # Lấy lịch sử chat từ repository
            items = chat_service.get_chat_history(session_id=session_id, limit=limit)
            
            # Format các mục thành "{role} : {content}"
            formatted_items = [f"{item[1]} : {item[0]}" for item in items]
            
            return BasicResponse(
                status="Success",
                message="Retrieved message history successfully",
                data=formatted_items
            )
        except Exception as e:
            self.logger.error(f"Failed to get message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to get message history: {str(e)}",
                data=None
            )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin của một chat session
        
        Args:
            session_id: ID của chat session
            
        Returns:
            Optional[Dict[str, Any]]: Thông tin session hoặc None nếu không tìm thấy
        """
        try:
            from src.database.db_connection import db
            
            with db.session_scope() as session:
                chat_session = session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).first()
                
                if not chat_session:
                    return None
                    
                return {
                    "id": str(chat_session.id),
                    "user_id": chat_session.user_id,
                    "organization_id": chat_session.organization_id,
                    "title": chat_session.title,
                    "start_date": chat_session.start_date
                }
        except Exception as e:
            self.logger.error(f"Failed to get session info: {str(e)}")
            return None
                          
    def delete_message_history(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Delete the chat history for a session
        
        Args:
            session_id: The ID of the chat session to delete
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response indicating success or failure
        """
        try:
            # Kiểm tra quyền xóa nếu cung cấp user_id
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Kiểm tra xem user_id có khớp với session owner không
                    if session_info.get("user_id") != user_id:
                        # Kiểm tra xem user có quyền admin trong tổ chức không
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Cần kiểm tra vai trò admin ở đây nếu có thể
                            # import tạm user_role_service
                            from src.handlers.user_role_handler import UserRoleService
                            user_role_service = UserRoleService()
                            is_admin = user_role_service.is_admin(user_id, organization_id)
                            if not is_admin:
                                return BasicResponse(
                                    status="Failed",
                                    message="You don't have permission to delete this chat history",
                                    data=None
                                )
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to delete this chat history",
                                data=None
                            )
            
            if chat_service.is_session_exist(session_id):
                chat_service.delete_chat_history(session_id=session_id)
                return BasicResponse(
                    status="Success",
                    message="Chat history deleted successfully",
                    data=None
                )
            else:
                return BasicResponse(
                    status="Failed",
                    message="Chat session does not exist",
                    data=None
                )
        except Exception as e:
            self.logger.error(f"Failed to delete message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to delete message history: {str(e)}",
                data=None
            )