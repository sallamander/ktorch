"""py.test configuration file"""

import pytest


def pytest_addoption(parser):
    """Add custom py.test flag options"""

    parser.addoption(
        '--skip_gpu_tests', action='store_true',
        help=(
            'Skip all tests that are earmarked as specifically testing '
            'the use of GPU functionality.'
        )
    )


def pytest_collection_modifyitems(config, items):
    """"""

    if not config.getoption('--skip_gpu_tests'):
        return

    skip_gpu_tests_marker = pytest.mark.skip(reason=(
        '--skip_gpu_tests flag was passed, and this test or set of tests '
        'has been earmarked as specifically testing the use of GPU '
        'functionality.'
    ))
    for item in items:
        if 'skip_gpu_tests' in item.keywords:
            item.add_marker(skip_gpu_tests_marker)
