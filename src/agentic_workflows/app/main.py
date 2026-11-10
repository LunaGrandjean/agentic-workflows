"""Run with: chainlit run src/agentic_workflows/app/main.py"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chainlit as cl

from agentic_workflows.constants import WORKFLOW_TASK
from agentic_workflows.online_prediction import PredictionError, describe_restaurant, translate_sentence

logger = logging.getLogger(__name__)


@cl.on_chat_start
async def start() -> None:
    if WORKFLOW_TASK == "restaurant":
        await cl.Message(content=("Send restaurant attributes to generate an English description.\n\n"
            "Example: `name[The Eagle], eatType[restaurant], food[Italian], priceRange[moderate], "
            "area[riverside], familyFriendly[yes]`\n\n"
            "Available fields: name, eatType, food, priceRange, customer rating, area, familyFriendly, near."), author="Restaurant Studio").send()
    else:
        await cl.Message(content="Send an English sentence, and I’ll rewrite it in Yoda’s style.", author="Yoda").send()


@cl.on_message
async def main(message: cl.Message) -> None:
    try:
        generate = describe_restaurant if WORKFLOW_TASK == "restaurant" else translate_sentence
        answer = await cl.make_async(generate)(message.content)
    except PredictionError as exc:
        answer = str(exc)
    except Exception:
        logger.exception("Online prediction failed")
        answer = "The generation service is unavailable. Please try again shortly."
    await cl.Message(content=answer, author="Restaurant Studio" if WORKFLOW_TASK == "restaurant" else "Yoda").send()
