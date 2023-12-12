import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

only_special_chars = "[а-яёА-ЯЁa-zA-Z0-9]"
greetings_regexp = (
    "((п|П)ривет)|((д|Д)обрый день)|((к|К)укусики)|((к|К)уку)|((д|Д)оброе утро)|((д|Д)обрый день)"
)

BOT_ROLE = "bot"
USER_ROLE = "user"


@dataclass
class Message:
    role: str
    content: str


class BaseSocialReader(ABC):
    r"""Base class for reading from different data source.

    For each source create own implementation with overriding all abstract methods
    """
    data_path: str
    filtered_chats: list[list[Message]]

    def __init__(self, data_path: str, num_workers: int = 8, debug: bool = False):
        r"""
        Args:
            data_path: path to root of raw data
            num_workers: number of threads using for tokenize and process data
            debug: flag for reducing number of samples for debug your algo
        """
        self.data_path = data_path
        self.num_workers = num_workers
        self.debug = debug

    @abstractmethod
    def prepare_data(self) -> list[list[Message]]:
        r"""Read data and return in proper format."""

    @abstractmethod
    def is_my_message(self, message: dict) -> bool:
        """
        Is message sent by current TG user
        Args:
            message: telegram message
        """

    def get_role_from_message(self, message: dict) -> str:
        if self.is_my_message(message):
            role = BOT_ROLE
        else:
            role = USER_ROLE
        return role

    def my_message_labels(self, messages: list[dict]) -> list[Message]:
        """
        Is message sent by current TG user
        Args:
            messages: telegram message
        """
        results = []
        for message in messages:
            role = self.get_role_from_message(message)
            results.append(Message(role=role, content=message["text"]))
        return results

    @staticmethod
    def remove_user_message_by_cond(messages: list[dict]) -> list[dict]:
        r"""
        Remove some user message by condition
        Args:
            messages: user messages
        """
        filtered_data = []
        for message in messages:
            if "анекдот" in message["text"] or "Анекдот" in message["text"]:
                continue

            if not len(re.findall(only_special_chars, message["text"])):
                continue

            filtered_data.append(message)
        return filtered_data

    @staticmethod
    def split_chat_by_greetings(messages: list[Message]):
        r"""Split user dialog by greetings.

        When we meet greeting, we remove all previous context message and start new dialog
        Args:
            messages: user messages
        """
        new_chats = []
        current_chat = []
        for message in messages:
            if message.role == "user" and re.findall(greetings_regexp, message.content):
                if current_chat:
                    new_chats.append(current_chat)
                current_chat = [message]
            else:
                current_chat.append(message)
        if len(current_chat) > 1:
            new_chats.append(current_chat)
        return new_chats
