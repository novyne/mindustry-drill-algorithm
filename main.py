import numpy as np

from collections import deque
from heapq import heappush, heappop
from itertools import combinations
from typing import Iterable, Optional

from ore import Ore


class DrillAlgorithm:
    
    def __init__(self, drillx: int, drilly: int, ore: Ore, drillmap: Optional[np.ndarray]=None) -> None:
        """Drill algorithm class. Determines a (near) optimal placement for drills with a given x and y dimension on a given ore.
        Args:
            drillx (int): The x dimension of the drill.
            drilly (int): The y dimension of the drill.
            ore (Ore): The ore for the drills to be placed on.
            drillmap (np.ndarray): An optional drillmap.
        """
        
        self.drillx = drillx
        self.drilly = drilly
        self.ore = ore
        
        self.x = self.ore.x
        self.y = self.ore.y
        
        if drillmap is None:
            self.drillmap = np.zeros((self.x, self.y), dtype=int)
        else:
            self.drillmap = drillmap
        self.placed_drills = []
        
        self.conveyormap = np.zeros((self.x, self.y), dtype=int)
    
    def __setitem__(self, key, value):
        self.drillmap[key] = value
    def __getitem__(self, key):
        return self.drillmap[key]
    
    def __str__(self) -> str:
        s = ''
        
        # charset = '█+X '
        charset = ['\033[92m█','\033[94m█','\033[91m█',' ', '\033[93m█','\033[95m']
        
        for x in range(self.x):
            for y in range(self.y):
                if self.drillmap[x, y] == 1 and self.conveyormap[x, y] == 1:
                    i = 5
                if self.drillmap[x, y] == 1 and self.ore[x, y] == 1:
                    i = 0
                elif self[x, y] == 1:
                    i = 2
                elif self.conveyormap[x, y] == 1:
                    i = 4
                elif self.ore[x, y] == 1:
                    i = 1
                else:
                    i = 3
                s += charset[i]
            s += '\n'
        s += '\033[0m\n'
        return s
    
    def place_drill(self, x: int, y: int) -> None:
        """Place a drill at the given coordinate, anchored north-west.
        Args:
            x (int): The x coordinate of the placement.
            y (int): The y coordinate of the placement.
        """
        
        for dx in range(self.drillx):
            for dy in range(self.drilly):
                self[x + dx, y + dy] = 1
        self.placed_drills.append((x, y))
        
    def remove_drill(self, x: int, y: int) -> None:
        """Remove the drill at the given coordinate, anchored north-west.
        Args:
            x (int): The x coordinate of the removal.
            y (int): The y coordinate of the removal.
        """
        
        for dx in range(self.drillx):
            for dy in range(self.drilly):
                self[x + dx, y + dy] = 0
        self.placed_drills.remove((x, y))
        
    def isplaceable(self, x: int, y: int) -> bool:
        """Determine whether a drill can be placed at the given coordinate, anchored north-west.
        Args:
            x (int): The x coordinate of the placement.
            y (int): The y coordinate of the placement.
        Returns:
            bool: True if the drill can be placed, otherwise False.
        """
        
        for dx in range(self.drillx):
            for dy in range(self.drilly):
                rx, ry = x + dx, y + dy
                
                # out of range
                if rx not in range(self.x) or ry not in range(self.y):
                    return False
                
                # if space is occupied by a drill
                if self[rx, ry] == 1:
                    return False
                
                # if the space has ore in
                if self.ore[rx, ry] == 1:
                    return True
        return False    
        
    def get_yield(self, x: int, y: int) -> int:
        """Get the number of ore squares covered by a drill placed at x y, anchored north-west.
        Args:
            x (int): The x coordinate of the drill.
            y (int): The y coordinate of the drill.
        Returns:
            int: The number of ore squares covered by the drill.
        """
        
        ore_covered = 1
        
        for dx in range(self.drillx):
            for dy in range(self.drilly):
                if self.ore[x + dx, y + dy] == 1:
                    ore_covered += 1
        return ore_covered
        
    def get_all_possible_placements(self) -> Iterable[tuple[int,int]]:
        """Obtain all the possible placements for a drill without obstructing the escape path.
        Returns:
            Iterable(tuple(int,int)): The coordinates of possible drill placements.
        """
        
        for x in range(self.x):
            for y in range(self.y):
                if self.isplaceable(x, y) and self.can_be_placed_without_obstruction(x, y):
                    yield x, y

    def all_drills_have_escape(self) -> bool:
        """Check if every drill-covered square has a path out.
        Returns:
            bool: True if every drill-covered square has a path out, otherwise False.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        queue = deque()
        visited = set()
        reachable_drills = set()
        all_drill_squares = set()

        # find all drill-covered squares and initialize BFS from adjacent open spaces
        for x in range(self.x):
            for y in range(self.y):
                if self.drillmap[x, y] == 1:
                    all_drill_squares.add((x, y))
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.x and 0 <= ny < self.y and self.drillmap[nx, ny] == 0:
                            queue.append((nx, ny))
                            visited.add((nx, ny))

        # BFS to find reachable drill squares
        while queue:
            cx, cy = queue.popleft()

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy

                if 0 <= nx < self.x and 0 <= ny < self.y and (nx, ny) not in visited:
                    if self.drillmap[nx, ny] == 0:  # open space
                        queue.append((nx, ny))
                        visited.add((nx, ny))
                    elif (nx, ny) in all_drill_squares:  # drill square reached
                        reachable_drills.add((nx, ny))
                        if reachable_drills == all_drill_squares:
                            return True  # early termination if all drills are reachable

        return reachable_drills == all_drill_squares

    def can_be_placed_without_obstruction(self, x: int, y: int) -> bool:
        """Test a drill placement at the given coordinates and see if all drills have an escape.
        Args:
            x (int): The x coordinate of the drill.
            y (int): The y coordinate of the drill.
        Returns:
            bool: Whether the drills have an escape or not.
        """
        
        test = DrillAlgorithm(self.drillx, self.drilly, self.ore, self.drillmap.copy())
        test.place_drill(x, y)
        return test.all_drills_have_escape()
    
    def place_max_safe_drills(self) -> None:
        """Place as many legal drills as possible which keeping the path intact."""
        
        while True:
            placements = list(self.get_all_possible_placements())
            if not placements:
                break
            
            # sort placements by yield
            placements.sort(key=lambda p: self.get_yield(p[0], p[1]), reverse=True)
            
            x, y = placements.pop(0)
            self.place_drill(x, y)

    def find_conveyor_path(self) -> None:
        """Find a path joining each drill together using MST and add it to the conveyor map."""
        # Directions for moving in 4-connected grid (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Step 1: Find all drill locations (anchored at the top-left corner of each drill)
        drill_locations = list(self.placed_drills)  # Use the stored drill positions

        if not drill_locations:
            return  # No drills to connect

        # Step 2: Use Prim's algorithm to find the minimum spanning tree (MST)
        mst = []  # Stores the edges of the MST
        visited = set()  # Tracks visited drills
        heap = []  # Priority queue for edges

        # Start with the first drill
        start = drill_locations[0]
        visited.add(start)

        # Add edges from the start drill to the heap
        for i in range(1, len(drill_locations)):
            end = drill_locations[i]
            distance = abs(start[0] - end[0]) + abs(start[1] - end[1])  # Manhattan distance
            heappush(heap, (distance, start, end))

        # Build the MST
        while heap and len(visited) < len(drill_locations):
            distance, start, end = heappop(heap)
            if end not in visited:
                visited.add(end)
                mst.append((start, end))

                # Add new edges to the heap
                for drill in drill_locations:
                    if drill not in visited:
                        distance = abs(end[0] - drill[0]) + abs(end[1] - drill[1])
                        heappush(heap, (distance, end, drill))

        # Step 3: Connect drills in the MST using BFS
        for start, end in mst:
            queue = deque()
            queue.append((start, [start]))  # (current_position, path)
            visited_bfs = set()
            visited_bfs.add(start)

            while queue:
                (cx, cy), path = queue.popleft()

                # Check if we've reached the end drill
                if (cx, cy) == end:
                    # Mark the path on the conveyor_map
                    for px, py in path:
                        self.conveyormap[px, py] = 1
                    break

                # Explore neighbors
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.x and 0 <= ny < self.y and (nx, ny) not in visited_bfs:
                        # Allow moving through open spaces or to the next drill
                        if self.drillmap[nx, ny] == 0 or (nx, ny) == end:
                            queue.append(((nx, ny), path + [(nx, ny)]))
                            visited_bfs.add((nx, ny))

                            # Skip the rest of the drill's rectangle once a drill is reached
                            if (nx, ny) == end:
                                break
        

    def score(self) -> float:
        """Score the drill placement pattern based on drill coverage and exposed ore.
        Returns:
            float: The score.
        """
        
        exposed_ore_squares = np.count_nonzero(~self.drillmap & self.ore.bitmap)
        empty_drill_squares = np.count_nonzero(self.drillmap & ~self.ore.bitmap)
        
        return -exposed_ore_squares ** 3 - empty_drill_squares ** 1.5


def main() -> None:
    """The main program."""
    
    ore = Ore((20, 20)).random(density=0.5)
    
    print(ore)
    
    drill = DrillAlgorithm(2, 2, ore)
    drill.place_max_safe_drills()
    drill.find_conveyor_path()
    print(drill)
    print(drill.score())
    
if __name__ == '__main__':
    main()