"""Generic CRUD operations base class for all database models."""

from typing import Any, Generic, Optional, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from finance_ai.database.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseCRUD(Generic[ModelType]):
    """
    Generic CRUD operations for SQLAlchemy models.

    Args:
        model: The SQLAlchemy model class to perform operations on.

    Example:
        >>> crud = BaseCRUD(User)
        >>> user = crud.get_by_id(session, "some-uuid")
    """

    def __init__(self, model: type[ModelType]) -> None:
        """Initialize with the target model class."""
        self.model = model

    def get_by_id(self, session: Session, record_id: str) -> Optional[ModelType]:
        """
        Get a single record by its primary key.

        Args:
            session: Database session.
            record_id: UUID string of the record.

        Returns:
            The model instance or None if not found.
        """
        return session.get(self.model, record_id)

    def get_all(self, session: Session, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """
        Get multiple records with pagination.

        Args:
            session: Database session.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            List of model instances.
        """
        statement = select(self.model).offset(skip).limit(limit)
        result = session.execute(statement)
        return list(result.scalars().all())

    def create(self, session: Session, **kwargs: Any) -> ModelType:
        """
        Create a new record. Caller must commit.

        Args:
            session: Database session.
            **kwargs: Field values for the new record.

        Returns:
            The created model instance.
        """
        instance = self.model(**kwargs)
        session.add(instance)
        session.flush()
        return instance

    def update(self, session: Session, record_id: str, **kwargs: Any) -> Optional[ModelType]:
        """
        Update an existing record by ID. Caller must commit.

        Args:
            session: Database session.
            record_id: UUID string of the record.
            **kwargs: Fields to update.

        Returns:
            The updated model instance or None if not found.
        """
        instance = self.get_by_id(session, record_id)
        if instance is None:
            return None
        for key, value in kwargs.items():
            setattr(instance, key, value)
        session.flush()
        return instance

    def delete(self, session: Session, record_id: str) -> bool:
        """
        Delete a record by ID. Caller must commit.

        Args:
            session: Database session.
            record_id: UUID string of the record.

        Returns:
            True if deleted, False if not found.
        """
        instance = self.get_by_id(session, record_id)
        if instance is None:
            return False
        session.delete(instance)
        session.flush()
        return True
