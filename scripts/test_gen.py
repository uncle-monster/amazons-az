from backend.engine.board import Board, P1, P2
from backend.engine.rules import generate_legal_actions

b = Board.initial()

a1 = generate_legal_actions(b, P1)
a2 = generate_legal_actions(b, P2)

print("P1 legal actions:", len(a1))
print("P2 legal actions:", len(a2))

# print first few actions
for i, a in enumerate(a1[:5]):
    print(i, a)