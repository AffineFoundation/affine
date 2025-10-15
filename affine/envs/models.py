from pydantic import BaseModel
from abc import ABC


class BaseEnv(BaseModel, ABC):
    class Config: arbitrary_types_allowed = True
    @property
    def name(self) -> str: return f"affine:{self.__class__.__name__.lower()}"
    def __hash__(self):     return hash(self.name)
    def __repr__(self):     return self.name
