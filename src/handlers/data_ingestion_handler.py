import os
import hashlib
import tempfile
import uuid
import aiofiles
import aiohttp
from urllib.parse import urlparse
from typing import Tuple, Optional, Dict, Any
from fastapi import UploadFile

from src.utils.config import settings
from src.helpers.qdrant_connection_helper import QdrantConnection
from src.database.data_layer_access.file_management_dal import FileManagementDAL

from src.handlers.file_partition_handler import DocumentExtraction
from src.utils.logger.custom_logging import LoggerMixin

# Temporarily disable FileManagementService
file_management = FileManagementDAL()


# Utils func      
async def read_file_from_path(path: str) -> bytes:
    normalized_path = path.replace('\\', '/')
    
    if ':' in normalized_path:
        drive, rest = normalized_path.split(':', 1)
        container_path = f"/app/data{rest}"
    else:
        container_path = normalized_path
    
    if not os.path.exists(container_path):
        alternative_paths = [
            f"/mnt/{normalized_path[0].lower()}{rest}" if ':' in normalized_path else None,
            os.path.join(os.getcwd(), os.path.basename(normalized_path)),
        ]
        
        for alt_path in alternative_paths:
            if alt_path and os.path.exists(alt_path):
                container_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Unable to find file '{path}' in container. "
                f"Please check the path and volume mount configuration again. "
                f"Tried paths: {container_path}, {', '.join(p for p in alternative_paths if p)}"
            )
    
    async with aiofiles.open(container_path, 'rb') as f:
        return await f.read()

async def get_file_data(file_path_or_url: str) -> bytes:
    parsed = urlparse(file_path_or_url)
    
    if parsed.scheme in ['http', 'https']:
        from aiohttp import ClientSession
        async with ClientSession() as session:
            async with session.get(file_path_or_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download file from {file_path_or_url}, status code: {response.status}")
                return await response.read()
    else:
        return await read_file_from_path(file_path_or_url)
    

class MockUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        
    async def read(self):
        return self._content
        
    async def seek(self, offset: int):
        pass


class DataIngestion(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()
        
        # Comment out vector database connection
        self.qdrant_client = QdrantConnection()
        self.data_extraction = DocumentExtraction() 
    
    @staticmethod
    def _save_temp_file(file_name: str, file_data: bytes) -> str:
        # Save the uploaded file to a temporary directory -> optimize for I/O
        TEMP_DIR = tempfile.gettempdir()
        temp_file_path = os.path.join(TEMP_DIR, file_name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_data)
        return temp_file_path

    async def ingest(self, file_or_url: UploadFile, collection_name: str, backend: str, organization_id: str, user_id: str, filename: Optional[str] = None) -> dict:
        self.logger.info('event=extract-metadata-from-file message="Ingesting document ..."')
        try:
            original_url = file_or_url
            
            if isinstance(file_or_url, str):
                file_url = file_or_url

                if filename:
                    actual_filename = filename
                else:
                    parsed = urlparse(file_url)
                    if parsed.scheme in ['http', 'https']:
                        actual_filename = os.path.basename(parsed.path)
                    else:
                        actual_filename = os.path.basename(file_url)
                
                
                file_extension = os.path.splitext(file_url)[1][1:]
                
                self.logger.info(f"Getting file from: {file_url}")
                try:
                    file_data = await get_file_data(file_url)
                    self.logger.info(f"Successfully got file: {actual_filename}, size: {len(file_data)} bytes")
                except Exception as e:
                    self.logger.error(f"Error getting file from {file_url}: {str(e)}")
                    return {"status": "error", "message": f"Error getting file: {str(e)}", "data": None}
            
                file = MockUploadFile(filename=actual_filename, content=file_data)

            else:
                file = file_or_url
                file_data = await file.read()
                file_extension = os.path.splitext(file_or_url)[1][1:]
                self.logger.info(f"file_extension: {file_extension}")
                await file.seek(0) 
                original_url = file_or_url

            temp_file_path = self._save_temp_file(file_name=file.filename, file_data=file_data)
            self.logger.info(f"temp_file_path: {temp_file_path}")

            # Calculate hash
            sha256 = hashlib.sha256(file_data).hexdigest()
            
            # Generate a UUID instead of using file management service
            document_id = str(uuid.uuid4())
            
            # Print file metadata for debugging
            print(f"File Metadata:")
            print(f"  - Name: {file.filename}")
            print(f"  - Collection: {collection_name}")
            print(f"  - Extension: {file_extension}")
            print(f"  - Size: {len(file_data)} bytes")
            print(f"  - SHA256: {sha256}")
            print(f"  - Document ID: {document_id}")
            
            # Extract text from the file
            resp = await self.data_extraction.extract_text(
                file=file,
                backend=backend,
                temp_file_path=temp_file_path, 
                document_id=document_id
            )
            
            self.logger.info(f"resp.status: {resp.status}")
            
            if resp.status == "success" and resp.data:
                file_management.create_file_record(
                    document_id=document_id,
                    file_name=file.filename,
                    extension=file_extension,
                    file_url=original_url,
                    created_by=user_id,
                    size=len(file_data),
                    sha256=sha256,
                    collection_name=collection_name,
                    organization_id=organization_id
                )

                # Comment out adding to vector database
                await self.qdrant_client.add_data(
                    documents=resp.data, 
                    collection_name=collection_name,
                    organization_id=organization_id
                )
                
                self.logger.info(f"Chunking Results:")
                self.logger.info(f"  - Total chunks: {len(resp.data)}")
                
                data = [docs.json() for docs in resp.data]
                return {"status": "success", "message": "Processed successfully", "data": data}
            else:
                return {"status": "failed", "message": "No content extracted from document", "data": None}
        
        except Exception as e:
            self.logger.error(f'error={str(e)}')
            return {"status": "error"}