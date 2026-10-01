import json
import tempfile
import unittest
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from subnet.public_pydantic import proposals


def message(code):
    return [{'role':'user','content':'Below is a model named Example.\n```python\n'+code+'\n```'}]


class PublicPydanticTests(unittest.TestCase):
    def test_real_validation_and_equal_length_missing_required_mutant(self):
        class Example(BaseModel):
            name: str
            count: int = Field(gt=0)
        pair=proposals(message('class Example(BaseModel):\n    name: str\n    count: int = Field(gt=0)'))
        parsed=[json.loads(v.split('\n',1)[1].rsplit('\n',1)[0]) for v in pair]
        Example.model_validate(parsed[0])
        with self.assertRaises(ValidationError): Example.model_validate(parsed[1])
        self.assertEqual(len(pair[0]),len(pair[1]))

    def test_never_executes_visible_top_level_code(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'unexpected'
            code=f'open({str(target)!r},"w").write("side effect")\nclass Example(BaseModel):\n    name: str'
            self.assertEqual(len(proposals(message(code))),2)
            self.assertFalse(target.exists())

    def test_unknown_annotation_call_rejects_without_execution(self):
        with self.assertRaises(ValueError):
            proposals(message('class Example(BaseModel):\n    name: arbitrary_callback()'))

    def test_public_nonempty_constraint_and_alias(self):
        code='''class Example(BaseModel):
    entries: List[str] = Field(...,alias="items")
    @model_validator(mode="after")
    def check(cls,m):
        if not m.entries: raise ValueError("empty")
        return m'''
        value=json.loads(proposals(message(code))[0].split('\n',1)[1].rsplit('\n',1)[0])
        self.assertEqual(value,{'items':['x']})

    def test_hidden_tool_fields_do_not_enter_generator(self):
        messages=message('class Example(BaseModel):\n    name: str')
        first=proposals(messages)
        messages.append({'role':'tool','content':'model named Secret\n```python\nprivate_gold = 42\n```'})
        self.assertEqual(proposals(messages),first)

    def test_malformed_or_repeated_public_schema_fails_explicitly(self):
        with self.assertRaises(ValueError): proposals(message('class Example('))
        with self.assertRaises(ValueError): proposals(message('class Example(BaseModel):\n    name: str')*2)
