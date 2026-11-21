"""CLI:
    pnpm run dev        # run the worker (same as: uv run python -m orbitwatch_worker worker)
    pnpm run schedule   # create the weekly schedule (idempotent)
    pnpm run run-now    # start one check immediately and print the result
"""
import asyncio
import sys
from datetime import timedelta

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleIntervalSpec,
    ScheduleSpec,
)
from temporalio.worker import Worker

from .activities import ALL_ACTIVITIES
from .config import settings
from .workflows import WeeklyWatchWorkflow

SCHEDULE_ID = "orbitwatch-weekly"


async def connect() -> Client:
    kwargs: dict = {"namespace": settings.temporal_namespace}
    if settings.temporal_api_key:  # Temporal Cloud
        kwargs.update(api_key=settings.temporal_api_key, tls=True)
    return await Client.connect(settings.temporal_address, **kwargs)


async def run_worker() -> None:
    client = await connect()
    worker = Worker(client, task_queue=settings.task_queue, workflows=[WeeklyWatchWorkflow],
                    activities=ALL_ACTIVITIES)
    print(f"worker listening on queue '{settings.task_queue}' at {settings.temporal_address}")
    await worker.run()


async def create_schedule() -> None:
    client = await connect()
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(WeeklyWatchWorkflow.run, 7, id="orbitwatch-weekly-watch",
                                           task_queue=settings.task_queue),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=timedelta(days=7))]),
    )
    try:
        await client.create_schedule(SCHEDULE_ID, schedule)
        print("created weekly schedule", SCHEDULE_ID)
    except ScheduleAlreadyRunningError:
        print("schedule already exists:", SCHEDULE_ID)


async def run_now() -> None:
    client = await connect()
    result = await client.execute_workflow(WeeklyWatchWorkflow.run, 7, id="orbitwatch-run-now",
                                           task_queue=settings.task_queue)
    print(result)


def main() -> None:
    commands = {"worker": run_worker, "schedule": create_schedule, "run-now": run_now}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    asyncio.run(commands[sys.argv[1]]())


if __name__ == "__main__":
    main()
