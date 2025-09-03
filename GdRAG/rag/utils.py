from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from typing import List


class SingleFileSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return [text]

class LineBreakTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return text.split("\n\n")