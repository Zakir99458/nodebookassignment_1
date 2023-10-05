import random, math
def estimate_circle_area(k):
     R = 2
     niters = k
     total, in_circle = 0, 0
     for j in range(20):
         for i in range(niters):
             x = random.uniform(-R, R)
             y = random.uniform(-R, R)
             if x ** 2 + y ** 2 <= R ** 2:
                 in_circle += 1
             total += 1
         area = (2 * R) ** 2 * in_circle / total
         expected = math.pi * R ** 2
         print(f'After {(j + 1) * niters:>8} iterations area is {area:.08f}, ' +
             f'error is {abs(area - expected):.08f}', flush = True)
k_values = [10, 100, 1000, 10000]

for k in k_values:
    estimated_area = estimate_circle_area(k)
    # print(f"Estimated Area for k={k}: {estimated_area:.4f}")