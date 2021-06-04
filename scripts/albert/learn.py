import numpy as np

blocks = [ [0]*10 for i in range (5)]

# window = [ [0]*50 for i in range(50)]
window = np.zeros((50,50), dtype=np.uint8)

triple = [ [ window for j in range (10)] for i in range (5)]

print(window)
print(triple)

index = 0
# for i in range(5):
#     for j in range(10):
#         blocks[i][j] = index
#         index += 1

# print(blocks)