from abc import ABC, abstractmethod


class DataCut(ABC):
    def __init__(self, *, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def __call__(self, data_file, entry: dict) -> bool:
        if not self.enabled:
            return True
        return self.accepts(data_file, entry)

    def describe(self) -> str:
        return getattr(self, "cut_description", self.__doc__ or "").strip()

    @abstractmethod
    def accepts(self, data_file, entry: dict) -> bool:
        """Return True when this loaded entry passes the cut."""
        pass
