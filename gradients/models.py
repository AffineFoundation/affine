from pydantic import BaseModel


class SATData(BaseModel):
    prompt: str
    cls: list[list[int]]
    sol: dict[str, bool]


class ABDData(BaseModel):
    prompt: str
    program: str
    expected_output: str


class SATEnvironment(BaseModel):
    cls: list[list[int]]
    sol: dict[str, bool]


class ABDEnvironment(BaseModel):
    program: str
    expected_output: str


class EnvironmentData(BaseModel):
    sat: SATEnvironment | bool
    abd: ABDEnvironment | bool


class DatasetEntry(BaseModel):
    prompt: str
    env: EnvironmentData