"""Run with: chainlit run src/agentic_workflows/app/main.py"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chainlit as cl

from agentic_workflows.online_prediction import PredictionError, translate_sentence

logger = logging.getLogger(__name__)


@cl.on_chat_start
async def start() -> None:
    await cl.Message(content="Send an English sentence, and I’ll rewrite it in Yoda’s style.", author="Yoda").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    try:
        answer = await cl.make_async(translate_sentence)(message.content)
    except PredictionError as exc:
        answer = str(exc)
    except Exception:
        logger.exception("Online prediction failed")
        answer = "The translation service is unavailable. Please try again shortly."
    await cl.Message(content=answer, author="Yoda").send()
