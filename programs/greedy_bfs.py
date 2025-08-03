import heapq

graph = {
    'S': [('A', 1), ('G', 10)],
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3), ('G', 4)],
    'D': [('G', 2)],
    'G': []
}

heuristics = {'S': 7, 'A': 6, 'B': 5, 'C': 4, 'D': 3, 'G': 0}

def greedy_bfs(start, goal):
    visited = set()
    queue = [(heuristics[start], start)]
    
    while queue:
        _, node = heapq.heappop(queue)
        print(f"Visiting: {node}")
        if node == goal:
            print("Goal reached")
            return
        visited.add(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                heapq.heappush(queue, (heuristics[neighbor], neighbor))

greedy_bfs('S', 'G')
