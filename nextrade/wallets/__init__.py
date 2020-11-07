from .portfolio import Portfolio
#from .swap_portfolio import SwapPortfolio
from .wallet import Wallet


_registry = {}


def get(identifier: str) -> Portfolio:
    """Gets the `TradingStrategy` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingStrategy`

    Raises:
        KeyError: if identifier is not associated with any `TradingStrategy`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `TradingStrategy`.'.format(identifier))
    return _registry[identifier]