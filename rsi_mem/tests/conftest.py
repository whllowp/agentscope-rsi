# -*- coding: utf-8 -*-
"""pytest 共享夹具与离线假模型。"""
from __future__ import annotations

import pytest

from agentscope.credential import CredentialBase
from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase, ChatResponse, FinishedReason
from agentscope.model import ChatUsage

from rsi_mem.config import RSIMemConfig


class DummyCredential(CredentialBase):
    """占位凭据。"""


class DummyFormatter(FormatterBase):
    async def format(self, msgs):
        return [{"role": m.role, "content": m.get_text_content() or ""}
                for m in msgs]


class DummyChatModel(ChatModelBase):
    """离线假模型：按脚本回放固定回复，记录收到的消息供断言。"""

    def __init__(self, script: list[str] | None = None) -> None:
        super().__init__(
            credential=DummyCredential(),
            model="dummy-chat",
            parameters=self.Parameters(),
            stream=False,
        )
        self.formatter = DummyFormatter()
        self.script = list(script or [])
        self.received: list[list[Msg]] = []

    async def _call_api(self, model, messages, tools=None, tool_choice=None,
                        **kwargs):
        self.received.append(list(messages))
        text = (
            self.script.pop(0)
            if self.script
            else "(mock reply) 已根据记忆处理请求。"
        )
        return ChatResponse(
            content=[TextBlock(type="text", text=text)],
            is_last=True,
            finished_reason=FinishedReason.COMPLETED,
            usage=ChatUsage(input_tokens=8, output_tokens=4, time=0.01),
        )


@pytest.fixture
def config(tmp_path) -> RSIMemConfig:
    return RSIMemConfig(data_dir=str(tmp_path / ".rsi_mem"))
