from typing import Dict, Any

class ExecutionSimulator:
    """
    Models execution constraints based on the user's specific assumptions.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        # Cost to enter (implied probability)
        self.entry_price = 0.51
        # Target to exit unconditionally before expiration
        self.forced_exit_price = 0.99
    
    def attempt_entry(self, direction: str, candle: Dict[str, Any]) -> bool:
        """
        User defined assumption: 100% of 51c entry limit orders get filled.
        """
        return True

    def attempt_forced_exit(self, direction: str, candle: Dict[str, Any], final_outcome: str) -> bool:
        """
        User defined assumption: If the betting direction ends up being the final winning outcome 
        recorded in the dataset, we assume the 99c order got filled at some point before the end.
        """
        return direction == final_outcome

    def settle_trade(self, bet_direction: str, final_outcome: str, size: float, forced_exit_filled: bool) -> float:
        """
        Returns net PnL of the transaction.
        """
        cost = size * self.entry_price
        
        if forced_exit_filled:
            # We hit 99c before it expired
            revenue = size * self.forced_exit_price
            return revenue - cost
        
        if bet_direction == final_outcome:
            # Held to expiration and won
            revenue = size * 1.00
            return revenue - cost
        else:
            # Held to expiration and lost
            return -cost
