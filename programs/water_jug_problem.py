def water_jug_problem():
    maxA, maxB = 4, 3
    visited = set()
    stack = [(0, 0)]
    
    while stack:
        A, B = stack.pop()
        if (A, B) in visited:
            continue
        visited.add((A, B))
        print(f"A: {A}, B: {B}")
        if A == 2:
            print("Goal reached")
            return

        next_states = [
            (maxA, B), (A, maxB),
            (0, B), (A, 0),
            (min(A + B, maxA), max(0, B - (maxA - A))),
            (max(0, A - (maxB - B)), min(A + B, maxB))
        ]
        for state in next_states:
            if state not in visited:
                stack.append(state)

water_jug_problem()
