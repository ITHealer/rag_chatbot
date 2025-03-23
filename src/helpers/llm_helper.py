import re
import json
from langchain_community.chat_models import ChatOllama
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from typing import AsyncGenerator
from langchain_core.messages import AIMessage

class LLMGenerator(LoggerMixin):
    def __init__(self):
        super().__init__()


    def clean_thinking(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def get_llm(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        try:
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            # num_ctx=8000, 
                            streaming=True)
     
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
        return llm

    async def get_streaming_chain(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        """
        Get a configured LLM instance optimized for streaming responses
        
        This method is specifically designed for streaming use cases where
        chunks of text are returned incrementally rather than waiting for
        the complete response.
        
        Args:
            model: The name of the LLM model to use
            base_url: The base URL of the Ollama API
            
        Returns:
            ChatOllama: Configured LLM instance with streaming enabled
        """
        try:
            # For streaming, we use the same configuration as regular LLM
            # but ensure streaming is explicitly enabled
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            streaming=True)
            
            return llm
        except Exception as e:
            self.logger.error(f"Error configuring streaming LLM: {str(e)}")
            raise
    
    async def stream_response(self, 
                              llm,
                              messages,
                              clean_thinking: bool = True) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the LLM
        
        Args:
            llm: The LLM instance to use
            messages: The messages to send to the LLM
            clean_thinking: Whether to clean thinking sections from chunks
            
        Yields:
            str: Chunks of the response
        """
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessage) and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
            elif hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
    