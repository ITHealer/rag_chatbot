from typing import Optional, Dict, Type, Any
from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.database.repository.user_orm_repository import UserORMRepository

class RepositoryFactory(LoggerMixin):    
    # Map repository names with implementation classes
    _legacy_repositories: Dict[str, Type] = {
        "user": UserORMRepository,
        # Add other legacy repositories
    }
    
    _orm_repositories: Dict[str, Type] = {
        "user": UserORMRepository,
        # Add other ORM repositories when converted
    }
    
    @classmethod
    def get_repository(cls, repo_name: str, use_orm: Optional[bool] = None) -> Any:
        """
        Returns an instance of the repository.

        Args:
            repo_name: Repository name ('user', 'chat', etc.)
            use_orm: True to use ORM, False to use legacy, None to use default

        Returns:
            Repository instance
        """
        # Determine whether to use ORM or not
        if use_orm is None:
            # Read from config, default is False
            use_orm = getattr(settings, "USE_ORM_REPOSITORIES", False)
        
        # Select appropriate repository map
        repo_map = cls._orm_repositories if use_orm else cls._legacy_repositories
        
        # Check if repository exists
        if repo_name not in repo_map:
            cls().logger.error(f"Repository {repo_name} not found")
            raise ValueError(f"Repository {repo_name} not found")
        
        # Create and return instance
        return repo_map[repo_name]()