def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")

    for neighbor in graph.get(start, set()):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

# Example Usage:
# Create the graph
graph = {}
while True:
    try:
        nodes = int(input("Enter number of nodes: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

while True:
    try:
        edges = int(input("Enter number of edges: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

for i in range(edges):
    while True:
        try:
            n1, n2 = input(f"Enter edge {i+1} (node1 node2): ").split()
            graph.setdefault(n1, set()).add(n2)
            graph.setdefault(n2, set()).add(n1)
            break
        except ValueError:
            print("Invalid input. Please enter two nodes separated by a space.")

start_node = input("Enter the starting node for DFS: ")
print("DFS Traversal:")
dfs(graph, start_node)
