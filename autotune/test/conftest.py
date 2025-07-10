import time
from datetime import timedelta


def pytest_collection_modifyitems(session, config, items):
    if not items:
        return

    start_time = time.time()
    total_tests = len(items)
    completed = 0
    test_results = []

    def pytest_runtest_logreport(report):
        nonlocal completed
        if report.when == "call":
            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = total_tests - completed
            eta = avg_time * remaining

            # Record test result
            if report.passed:
                test_results.append(f"\033[32m.\033[0m")  # Green dot for pass
            elif report.failed:
                test_results.append(f"\033[31mF\033[0m")  # Red F for fail
            elif report.skipped:
                test_results.append(f"\033[33ms\033[0m")  # Yellow s for skip

            eta_str = str(timedelta(seconds=int(eta)))

            # Print all results so far and progress
            print(
                f"\rProgress: {completed}/{total_tests} "
                f"({(completed/total_tests)*100:.1f}%) "
                f"ETA: {eta_str} "
                f"{''.join(test_results[:-1])}",
                end="",
                flush=True,
            )

            if completed == total_tests:
                print()

    config.pluginmanager.register(type("ProgressReporter", (), {"pytest_runtest_logreport": pytest_runtest_logreport}))
