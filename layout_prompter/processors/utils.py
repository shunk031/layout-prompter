from typing import Dict, Type

from layout_prompter.configs import LayoutDatasetConfig
from layout_prompter.processors import (
    CompletionProcessor,
    ContentAwareProcessor,
    GenRelationProcessor,
    GenTypeProcessor,
    GenTypeSizeProcessor,
    RefinementProcessor,
    TaskProcessorMixin,
    TextToLayoutProcessor,
)
from layout_prompter.typehint import Task

PROCESSOR_MAP: Dict[Task, Type[TaskProcessorMixin]] = {
    "gen-t": GenTypeProcessor,
    "gen-ts": GenTypeSizeProcessor,
    "gen-r": GenRelationProcessor,
    "completion": CompletionProcessor,
    "refinement": RefinementProcessor,
    "content": ContentAwareProcessor,
    "text": TextToLayoutProcessor,
}


def create_task_processor(
    dataset_config: LayoutDatasetConfig,
    task: Task,
) -> TaskProcessorMixin:
    processor_cls: Type[TaskProcessorMixin] = PROCESSOR_MAP[task]
    processor = processor_cls(dataset_config=dataset_config)
    return processor
