import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from .activities import find_new_scenes, list_sites, post_alert

RETRY = RetryPolicy(initial_interval=timedelta(seconds=5), maximum_interval=timedelta(minutes=2),
                    maximum_attempts=5)
TIMEOUT = timedelta(minutes=2)


@workflow.defn
class WeeklyWatchWorkflow:
    """Once a week: for every watched site, look for new Sentinel-2 scenes and raise an alert."""

    @workflow.run
    async def run(self, days: int = 7) -> dict:
        since = (workflow.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sites = await workflow.execute_activity(list_sites, start_to_close_timeout=TIMEOUT, retry_policy=RETRY)

        results = await asyncio.gather(*[
            workflow.execute_activity(find_new_scenes, args=[site, since],
                                      start_to_close_timeout=TIMEOUT, retry_policy=RETRY)
            for site in sites
        ])
        fresh = [r for r in results if r["count"] > 0]
        for r in fresh:
            await workflow.execute_activity(post_alert, r, start_to_close_timeout=TIMEOUT, retry_policy=RETRY)

        return {"since": since, "sitesChecked": len(sites), "sitesWithNewScenes": [r["siteId"] for r in fresh]}
