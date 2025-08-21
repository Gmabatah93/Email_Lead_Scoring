"""
Types of Tests
1. Unit tests: tests on individual components that each have a single responsibility (ex. function that filters a list).
2. Integration tests: tests on the combined functionality of individual components (ex. data processing).
3. System tests: tests on the design of a system for expected outputs given inputs (ex. training, inference, etc.).
4. Acceptance tests: tests to verify that requirements have been met, usually referred to as User Acceptance Testing (UAT).
5. Regression tests: tests based on errors we've seen before to ensure new changes don't reintroduce them.


Best practices
Regardless of the framework we use, it's important to strongly tie testing into the development process.

- atomic: when creating functions and classes, we need to ensure that they have a single responsibility so that we can easily test them. If not, we'll need to split them into more granular components.
- compose: when we create new components, we want to compose tests to validate their functionality. It's a great way to ensure reliability and catch errors early on.
- reuse: we should maintain central repositories where core functionality is tested at the source and reused across many projects. This significantly reduces testing efforts for each new project's code base.
- regression: we want to account for new errors we come across with a regression test so we can ensure we don't reintroduce the same errors in the future.
- coverage: we want to ensure 100% coverage for our codebase. This doesn't mean writing a test for every single line of code but rather accounting for every single line.
- automate: in the event we forget to run our tests before committing to a repository, we want to auto run tests when we make changes to our codebase. We'll learn how to do this locally using pre-commit hooks and remotely via GitHub actions in subsequent lessons.

Implementation
tests/
├── code/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_predict.py
│   ├── test_train.py
│   ├── test_tune.py
│   ├── test_utils.py
│   └── utils.py
├── data/
│   ├── conftest.py
│   └── test_dataset.py
└── models/
│   ├── conftest.py
│   └── test_behavioral.py
"""
from typing import Iterable, Any, Dict, List

def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    return [index_to_class[index] for index in indices]
