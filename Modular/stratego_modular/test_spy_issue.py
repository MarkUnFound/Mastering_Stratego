from piece import PieceType
from board import LAKE_SQUARE

# Check the value of the Spy piece
spy_value = PieceType.SPY.value
print(f"Spy piece value: {spy_value}")
print(f"Lake square value: {LAKE_SQUARE}")

if spy_value == abs(LAKE_SQUARE):
    print("Issue found: Spy value conflicts with lake value!")
else:
    print("No conflict between Spy and lake values.")
