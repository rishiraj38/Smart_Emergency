import random
import heapq
from config import CITY

class CityGraph:
    def __init__(self, seed: int = None):
        if seed is not None:
            self.rng = random.Random(seed)
        else:
            self.rng = random.Random()
            
        # 20 Nodes
        self.nodes = [
            "NODE_INT1", "NODE_INT2", "NODE_INT3", "NODE_INT4", "NODE_INT5",
            "NODE_INT6", "NODE_INT7", "NODE_INT8", "NODE_INT9", "NODE_INT10",
            "NODE_H1", "NODE_H2",
            "NODE_RES1", "NODE_RES2", "NODE_RES3", "NODE_RES4", "NODE_RES5",
            "STATION_1", "STATION_2", "STATION_3"
        ]
        
        self.edges = {}
        self.blocked_nodes = set()
        self._generate_graph()
        
    def _generate_graph(self):
        # Generate connected graph
        for node in self.nodes:
            self.edges[node] = {}
            
        # Create a spanning tree first to ensure connectivity
        shuffled_nodes = list(self.nodes)
        self.rng.shuffle(shuffled_nodes)
        
        for i in range(1, len(shuffled_nodes)):
            u = shuffled_nodes[i]
            v = self.rng.choice(shuffled_nodes[:i])
            weight = self.rng.randint(1, 3)
            self.edges[u][v] = weight
            self.edges[v][u] = weight
            
        # Add random additional edges
        for _ in range(30):
            u = self.rng.choice(self.nodes)
            v = self.rng.choice(self.nodes)
            if u != v and v not in self.edges[u]:
                weight = self.rng.randint(1, 5)
                self.edges[u][v] = weight
                self.edges[v][u] = weight
                
    def block_node(self, node: str):
        self.blocked_nodes.add(node)
        
    def unblock_node(self, node: str):
        if node in self.blocked_nodes:
            self.blocked_nodes.remove(node)
            
    def shortest_path(self, start: str, end: str):
        if start == end:
            return [start]
            
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
        previous = {node: None for node in self.nodes}
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_dist > distances[current_node]:
                continue
                
            if current_node == end:
                break
                
            for neighbor, weight in self.edges[current_node].items():
                if neighbor in self.blocked_nodes and neighbor != end and neighbor != start:
                    continue # Cannot route through blocked nodes unless it's the destination/start
                    
                dist = current_dist + weight
                if dist < distances[neighbor]:
                    distances[neighbor] = dist
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (dist, neighbor))
                    
        if distances[end] == float('inf'):
            return None
            
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        return path
