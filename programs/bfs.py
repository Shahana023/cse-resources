# BFS Implementation

def bfs(graph, start):
    visited = set()
    queue = [(start, [start])]

    while queue:
        node, path = queue.pop(0)
        if node not in visited:
            print(node, end=" ")
            visited.add(node)

            for n in graph.get(node, set()):
                if n not in visited:
                    queue.append((n, path + [n]))


# Create the graph
graph = {}
nodes = int(input("Enter number of nodes: "))
edges = int(input("Enter number of edges: "))

for i in range(edges):
    while True:
        try:
            n1, n2 = input(f"Enter edge {i+1} (node1 node2): ").split()
            graph.setdefault(n1, set()).add(n2)
            graph.setdefault(n2, set()).add(n1)
            break
        except ValueError:
            print("Invalid input. Please enter two nodes separated by a space.")

start = input("Enter the starting node: ")
print("BFS Traversal:")
bfs(graph, start)
