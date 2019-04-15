import pytest
import strategy

mock_validator_object = None

def training_method_1():
    pass
def recursive_response_1(attempt_record, validator, **kwargs):
    return None

def test_1():
    incomplete_strategy = strategy.Strategy(None, recursive_response_1, mock_validator_object)
    with pytest.raises(AssertionError):
        incomplete_strategy.do_strategy()
