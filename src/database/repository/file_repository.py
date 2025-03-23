import psycopg2
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from src.database.repository.user_orm_repository import UserORMRepository
from src.helpers.qdrant_connection_helper import QdrantConnection
from src.utils.config import settings
from src.utils.constants import TypeDatabase
from src.utils.logger.custom_logging import LoggerMixin


class FileProcessingVecDB(LoggerMixin):
    def __init__(self):
        super().__init__()
        self.qdrant_client = QdrantConnection()

    async def delete_document_by_file_name(self, 
                     file_name: str,
                     type_db: str = TypeDatabase.Qdrant.value,
                     collection_name: str = settings.QDRANT_COLLECTION_NAME,  # Thêm param
                     organization_id: Optional[str] = None):
        if not file_name:
            self.logger.error('event=delete-document-by-file-name '
                            'message="Delete document by file name Failed. '
                            f'error="file_name is None. Please check your input again." ')
        else:
            self.logger.info('event=delete-document-in-vector-database '
                        'message="Start delete ..."')
            
            if type_db == TypeDatabase.Qdrant.value:
                if not self.qdrant_client.client.collection_exists(collection_name=collection_name):
                    self.logger.warning(f"Collection {collection_name} does not exist, skipping delete operation")
                    return
                    
                await self.qdrant_client.delete_document_by_file_name(
                    document_name=file_name, 
                    collection_name=collection_name,
                    organization_id=organization_id
                )
    
    async def delete_document_by_batch_ids(self, 
                     document_ids: list[str],
                     type_db: str = TypeDatabase.Qdrant.value,
                     collection_name: str = settings.QDRANT_COLLECTION_NAME,
                     organization_id: Optional[str] = None):
        
        if not document_ids:
            self.logger.error('event=delete-document-by-batch-ids '
                            'message="Delete document by batch ids Failed. '
                            f'error="document_ids is None. Please check your input again." ')
        else: 
            self.logger.info('event=delete-document-in-vector-database '
                        'message="Start delete ..."')
            
            if type_db == TypeDatabase.Qdrant.value:
                if not self.qdrant_client.client.collection_exists(collection_name=collection_name):
                    self.logger.warning(f"Collection {collection_name} does not exist, skipping delete operation")
                    return
                    
                await self.qdrant_client.delete_document_by_batch_ids(
                    document_ids=document_ids, 
                    collection_name=collection_name,
                    organization_id=organization_id
                )


class FileProcessingRepository(UserORMRepository):
    def create_file_records(self, file_name, extension, file_url, uploaded_by, size, sha256, collection_name='', organization_id=None):

        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            sql = """INSERT INTO documents  
                     (id, file_name, collection_name, extension, size, status, 
                      created_by, created_at, sha256, organization_id) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            
            id_ = str(uuid.uuid4())
            created_at = datetime.now()
            cursor.execute(sql, (id_, created_at, extension, file_name, file_url, "", size, 
                                 uploaded_by, sha256, organization_id, collection_name))
            conn.commit()
            return id_
        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()


    def check_duplicates(self, sha256, file_name, organization_id=None):
        """
        Check for duplicate files within an organization
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            if organization_id:
                sql = """SELECT id FROM documents 
                         WHERE sha256 = %s AND file_name = %s AND organization_id = %s"""
                cursor.execute(sql, (sha256, file_name, organization_id))
            else:
                sql = """SELECT id FROM documents WHERE sha256 = %s AND file_name = %s"""
                cursor.execute(sql, (sha256, file_name))
            
            result = cursor.fetchone()
            return result is not None
        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()


    def get_files_by_search_engine(self, key_word=None, extension=None, created_at=None, 
                                  limit=10, offset=0, organization_id=None):
        """
        Find files by multiple conditions, filter by organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            base_sql = """
                SELECT id, file_name, extension, size, created_by, collection_name, organization_id
                FROM documents
                WHERE 1=1
            """
            count_sql = """
                SELECT COUNT(id)
                FROM documents
                WHERE 1=1
            """
            filters = ""
            params = []
            
            if organization_id:
                filters += " AND organization_id = %s"
                params.append(organization_id)
            
            if key_word is not None:
                filters += " AND LOWER(file_name) LIKE LOWER(%s)"
                params.append(f"%{key_word}%")

            if extension is not None:
                filters += " AND extension LIKE %s"
                params.append(extension)

            if created_at is not None:
                filters += " AND DATE(created_at) = %s"
                params.append(created_at)
                
            count_sql += filters
            base_sql += filters 
            base_sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(count_sql, params[:-2] if params else [])
            total_count = cursor.fetchone()[0]
            
            cursor.execute(base_sql, params)
            results = cursor.fetchall()
            
            files = [dict(zip(('id', 'file_name', 'extension', 'size', 'created_by', 'collection_name'), r)) for r in results]
            
            return {
                'total_count': total_count,
                'files': files
                }
        
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error searching files: {str(error)}")
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()


    def delete_document_by_batch_ids(self, document_ids: list[str], organization_id=None):
        """
        Delete multiple documents by ID, filter by organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            if not document_ids:
                return None
                
            ids_list = ', '.join(['%s'] * len(document_ids))
            
            if organization_id:
                sql = f"DELETE FROM documents WHERE id IN ({ids_list}) AND organization_id = %s"
                params = document_ids + [organization_id]
            else:
                sql = f"DELETE FROM documents WHERE id IN ({ids_list})"
                params = document_ids
                
            cursor.execute(sql, tuple(params))
            conn.commit()
            return None
        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()


    def delete_document_by_file_name(self, file_name, organization_id=None):
        """
        Delete documents by file name, filter by organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            if organization_id:
                sql = """DELETE FROM documents WHERE file_name = %s AND organization_id = %s"""
                cursor.execute(sql, (file_name, organization_id))
            else:
                sql = """DELETE FROM documents WHERE file_name = %s"""
                cursor.execute(sql, (file_name,))
                
            conn.commit()
            return None
        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()


    def get_document_by_id(self, document_id, organization_id=None):
        """
        Get document information by ID, filter by organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            if organization_id:
                sql = """SELECT file_name, collection_name 
                         FROM documents 
                         WHERE id = %s AND organization_id = %s 
                         LIMIT 1"""
                cursor.execute(sql, (document_id, organization_id))
            else:
                sql = """SELECT file_name, collection_name 
                         FROM documents 
                         WHERE id = %s 
                         LIMIT 1"""
                cursor.execute(sql, (document_id,))
                
            result = cursor.fetchone()
            if result:
                return result[0], result[1], result[2]
            return None
        except (Exception, psycopg2.Error) as error:
            raise ValueError(error)
        finally:
            if conn:
                cursor.close()
                conn.close()
    

    def get_file_details_by_id(self, document_id) -> Optional[Dict[str, Any]]:
        """
        Get file details by ID, including organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            sql = """SELECT id, file_name, collection_name, extension, size, status, 
                     created_by, created_at, sha256, organization_id
                     FROM documents WHERE id = %s LIMIT 1"""
            cursor.execute(sql, (document_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'file_name': result[1],
                    'collection_name': result[2],
                    'extension': result[3],
                    'size': result[4],
                    'status': result[5],
                    'created_by': result[6],
                    'created_at': result[7],
                    'sha256': result[8],
                    'organization_id': result[9]
                }
            return None
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting file details by ID: {str(error)}")
            return None
        finally:
            if conn:
                cursor.close()
                conn.close()
                

    def get_file_details_by_name(self, file_name) -> Optional[Dict[str, Any]]:
        """
        Get file details by name, including organization_id
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            sql = """SELECT id, file_name, collection_name, extension, size, status, 
                     created_by, created_at, sha256, organization_id
                     FROM documents WHERE file_name = %s LIMIT 1"""
            cursor.execute(sql, (file_name,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'file_name': result[1],
                    'collection_name': result[2],
                    'extension': result[3],
                    'size': result[4],
                    'status': result[5],
                    'created_by': result[6],
                    'created_at': result[7],
                    'sha256': result[8],
                    'organization_id': result[9]
                }
            return None
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting file details by name: {str(error)}")
            return None
        finally:
            if conn:
                cursor.close()
                conn.close()
                

    def get_files_by_organization(self, organization_id, limit=100, offset=0) -> List[Dict[str, Any]]:
        """
        Get a list of files belonging to an organization
        """
        conn = self.create_connection()
        cursor = conn.cursor()
        try:
            sql = """SELECT id, file_name, collection_name, extension, size, status,
                     created_by, created_at, sha256
                     FROM documents
                     WHERE organization_id = %s
                     ORDER BY created_at DESC
                     LIMIT %s OFFSET %s"""
            cursor.execute(sql, (organization_id, limit, offset))
            
            results = cursor.fetchall()
            files = [dict(zip(('id', 'file_name', 'collection_name', 'extension', 'size', 
                               'status', 'created_by', 'created_at', 'sha256'), r)) 
                    for r in results]
            return files
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f"Error getting files by organization: {str(error)}")
            return []
        finally:
            if conn:
                cursor.close()
                conn.close()