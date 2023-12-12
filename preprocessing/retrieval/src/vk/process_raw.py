import logging
import os.path
from multiprocessing import Pool

from preprocessing.causal.vk.helper import (
    filter_function_chat_folder,
    filter_text,
    get_all_messages_by_user_path,
    get_user,
)
from preprocessing.retrieval.src.Retrieval import RetrievalMining
from tqdm import tqdm

from src.utils.logger import get_pylogger

log = get_pylogger(__name__)
logging.basicConfig(level=logging.INFO)


class VkRetrievalMining(RetrievalMining):
    def __init__(self, min_years: int, **kwargs):
        super().__init__(**kwargs)

        self.data_path = os.path.join(self.data_path, "messages")
        self.min_years = min_years

    def read_chats(self):
        all_chat_folder = get_user(self.data_path)
        filtered_chat_folders = list(filter(filter_function_chat_folder, all_chat_folder))

        if self.debug:
            filtered_chat_folders = filtered_chat_folders[::52]

        with Pool(self.num_workers) as p:
            total_messages = tqdm(
                p.imap(get_all_messages_by_user_path, filtered_chat_folders),
                total=len(filtered_chat_folders),
            )
            self.chats = list(total_messages)

    def filter_chats(self):
        total_messages_filtered_by_year = [
            remove_messages_by_year_thr(chat, self.min_years) for chat in self.chats
        ]
        total_messages_filtered_by_year = [
            user for user in total_messages_filtered_by_year if len(user)
        ]
        log.info(
            f"Filtered chat count remove old messages %s", len(total_messages_filtered_by_year)
        )

        total_messages_filtered_by_year_filtered_text = [
            filter_text(user) for user in total_messages_filtered_by_year
        ]
        self.filtered_chats = [
            user for user in total_messages_filtered_by_year_filtered_text if len(user)
        ]
