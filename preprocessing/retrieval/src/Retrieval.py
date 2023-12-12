import datasets
import torch
from preprocessing.utils.utils import write_jsonl
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer


class RetrievalMining:
    """Class helps to build dataset for retrieval task."""

    chats: list[list[dict]] = None
    filtered_chats: list[list[dict]] = None
    merged_message_chats: list[list[dict]] = None
    qa_dataset: list[dict]

    def __init__(
        self,
        data_path: str,
        output_path: str,
        num_workers: int = 8,
        debug: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.debug = debug
        self.output_path = output_path

        if debug:
            self.num_workers = 1
        else:
            self.num_workers = num_workers

    def read_chats(self):
        """Read all data."""
        pass

    def filter_chats(self):
        """Filter read chats."""
        pass

    def merge_messages_for_all_chats(self):
        """Merge messages and return qa list of data."""
        qa_chats = [self.merge_messages(chat) for chat in self.filtered_chats]
        self.qa_dataset = [item for sublist in qa_chats for item in sublist]
        return self.qa_dataset

    @staticmethod
    def merge_messages(df):
        df = list(filter(lambda x: len(x["content"].strip()), df))
        for idx, msg in enumerate(df):
            if msg["role"] == "user":
                df = df[idx:]
                break

        if len(df) < 2:
            return []

        first_sample = df[0]
        current_prefix = first_sample["role"]
        current_text = [first_sample["content"]]
        rows = []
        last_prefix = None
        for idx in range(1, len(df)):
            current_sample = df[idx]
            prefix = current_sample["role"]
            text = str(current_sample["content"])
            text = text.strip()
            last_prefix = prefix

            if prefix != current_prefix:
                rows.append({"role": current_prefix, "content": current_text})

                current_text = [current_sample["content"]]
                current_prefix = prefix
            else:
                current_text.append(text)

        if last_prefix == current_prefix:
            rows.append({"role": current_prefix, "content": current_text})

        total_len = 2 * (len(rows) // 2)

        qa_dataset = []

        for idx in range(0, total_len, 2):
            question_id = idx
            answer_id = idx + 1

            questions = "\n".join(rows[question_id]["content"])
            assert rows[question_id]["role"] == "user"

            answer_text = "\n".join(rows[answer_id]["content"])
            assert rows[answer_id]["role"] == "bot"
            qa_dataset.append({"question": questions, "answer": answer_text})
        return qa_dataset

    def run(self) -> list[dict]:
        self.read_chats()
        self.filter_chats()
        return self.merge_messages_for_all_chats()

    def save_results(self):
        write_jsonl(self.qa_dataset, self.output_path)


class FaissIndexCreator:
    def __init__(self, model_name: str, debug=False, device: str = "cpu"):
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
        self.debug = debug
        self.device = device
        self.model.to(device)
        self.model.eval()

    def prepare(self, qa_dataset: datasets.Dataset):
        if self.debug:
            qa_dataset = qa_dataset.train_test_split(test_size=0.01)["test"]

        with torch.no_grad():
            ds_with_embeddings = qa_dataset.map(
                lambda example: {
                    "embeddings": self.model(
                        **{
                            k: v.to(self.device)
                            for k, v in self.tokenizer(
                                example["question"],
                                return_tensors="pt",
                                max_length=512,
                                truncation=True,
                            ).items()
                        }
                    )[0][0]
                    .cpu()
                    .numpy()
                }
            )
            ds_with_embeddings.add_faiss_index(column="embeddings")
            return ds_with_embeddings
