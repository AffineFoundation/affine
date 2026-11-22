"""GSM8K-Platinum environment: framework doubles keep these tests CPU-only and offline."""
import importlib
from pathlib import Path
import sys
import types

import pytest


REVISED = {288: ("3", "4"), 403: ("81", "135"), 454: ("150", "240"), 649: ("162000", "160000"),
           749: ("3", "6"), 823: ("14", "18"), 952: ("360", "1800"), 1035: ("35", "40"),
           1071: ("251", "257"), 1309: ("2280", "2180")}
REMOVED = ({2} | set(range(4, 113))) - set(REVISED)  # 110 removed original questions; all ten revised ones stay
ORIGINAL = [{"question": f"question {i}", "answer": f"work\n#### {REVISED.get(i, (str(i), ''))[0]}"} for i in range(1319)]
PLATINUM = [{"question": row["question"], "answer": f"work\n#### {REVISED.get(i, ('', str(i)))[1]}",
             "cleaning_status": "revised" if i in REVISED else "verified"}
            for i, row in enumerate(ORIGINAL) if i not in REMOVED]


class Data:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_copy(self, update):
        return Data(**self.__dict__, **update)


class Task:
    def __init__(self, data, config):
        self.data, self.config = data, config


def _reward(**kw):
    return lambda method: method


@pytest.fixture
def taskset(monkeypatch):
    vf = types.SimpleNamespace(TaskData=Data, Task=Task, Trace=object, Runtime=object, reward=_reward)
    vf.__path__ = []
    vf.v1 = vf
    calls = []
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *a, **kw: (calls.append((a, kw)), PLATINUM)[1]
    verify = b"the pinned Hub verify.py bytes"
    hub = types.ModuleType("gsm8k.taskset")
    hub.GSM8KData, hub.GSM8KTaskset, hub.GSM8KTask = Data, type("GSM8KTaskset", (), {}), Task
    hub.SYSTEM, hub.VERIFY = "the unchanged Hub system prompt", verify
    gsm8k = types.ModuleType("gsm8k")
    gsm8k.GSM8KTaskset = hub.GSM8KTaskset
    monkeypatch.setitem(sys.modules, "gsm8k", gsm8k)
    monkeypatch.setitem(sys.modules, "verifiers", vf)
    monkeypatch.setitem(sys.modules, "verifiers.v1", vf)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "gsm8k.taskset", hub)
    for path in ("local-gsm8k", "local-gsm8k-platinum"):
        monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "environments" / path))
    for name in ("local_gsm8k_platinum", "local_gsm8k"):
        sys.modules.pop(name, None)
    local = importlib.import_module("local_gsm8k")
    platinum = importlib.import_module("local_gsm8k_platinum")
    return platinum, local, calls


def test_loads_pinned_dataset_in_file_order_with_hub_prompt_and_local_reward(taskset):
    platinum, local, calls = taskset
    instance = platinum.LocalGSM8KPlatinumTaskset()
    instance.config = types.SimpleNamespace(task="task config")
    tasks = list(instance.load())
    assert len(tasks) == 1209 and len(REMOVED) == 110
    assert calls == [(("madrylab/gsm8k-platinum", "main"),
                      {"split": "test", "revision": "e762492455a1cf7967de89f05b6bef72fc713b66"})]
    assert [t.data.idx for t in tasks] == list(range(1209))
    assert all(type(t) is local.LocalGSM8KTask and t.config == "task config" for t in tasks)
    assert platinum.LocalGSM8KTask.correct is local.LocalGSM8KTask.correct
    assert local.shorten("reasoning\n#### 42") == "reasoning\n#### 42"
    assert local.shorten("x" * (local.MAX_ARG + 1) + "\n#### 42 ") == "#### 42"
    assert local.shorten("x" * (local.MAX_ARG + 1)) is None
    assert all(t.data.prompt == f"the unchanged Hub system prompt\n\nquestion {i}"
               for t, i in zip(tasks, [i for i in range(1319) if i not in REMOVED]))
    assert all(t.data.network_allow == ["*"] and t.data.network_block == [] for t in tasks)
    answers = {i: tasks[j].data.answer for j, i in enumerate(i for i in range(1319) if i not in REMOVED)}
    assert {i: answers[i] for i in REVISED} == {i: new for i, (_, new) in REVISED.items()}
    assert tasks[0].data.answer == "0" and tasks[-1].data.answer == "1318"


def test_loader_import_is_lazy(taskset, monkeypatch):
    platinum, _, _ = taskset
    instance = platinum.LocalGSM8KPlatinumTaskset()
    instance.config = types.SimpleNamespace(task=None)
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ModuleNotFoundError):
        instance.load()
