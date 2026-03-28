"""CRUD operations for Personal Finance AI database models."""

from finance_ai.database.crud.base_crud import BaseCRUD
from finance_ai.database.crud.user_crud import UserCRUD
from finance_ai.database.crud.income_crud import IncomeCRUD
from finance_ai.database.crud.deduction_crud import DeductionCRUD
from finance_ai.database.crud.investment_crud import InvestmentHoldingCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.crud.tax_filing_crud import TaxFilingCRUD
from finance_ai.database.crud.financial_goal_crud import FinancialGoalCRUD
from finance_ai.database.crud.conversation_crud import ConversationCRUD
from finance_ai.database.crud.conversation_message_crud import ConversationMessageCRUD
from finance_ai.database.crud.schedule_crud import AssetScheduleCRUD, AssetNotificationCRUD

__all__ = [
    "BaseCRUD",
    "UserCRUD",
    "IncomeCRUD",
    "DeductionCRUD",
    "InvestmentHoldingCRUD",
    "TransactionCRUD",
    "TaxFilingCRUD",
    "FinancialGoalCRUD",
    "ConversationCRUD",
    "ConversationMessageCRUD",
    "AssetScheduleCRUD",
    "AssetNotificationCRUD",
]
