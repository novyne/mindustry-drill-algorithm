import numpy as np
import random as rn

from itertools import product
from typing import Optional


class Ore:
    
    def __init__(self, shape: tuple, *, bitmap: Optional[np.ndarray] = None) -> None:
        """Ore class. Stores ore placement in a numpy array with a padded border of 1 cell on each edge.
        Args:
            shape (tuple): The shape of the ore bitmap. Always 2D.
            bitmap (np.ndarray): A pre-existing bitmap. If provided, all previous parameters are disregarded.
        """
        
        if len(shape) != 2:
            raise ValueError(f"""Shape must be 2-dimensional. (Got {len(shape)} dimensions)""")
    
        for i, dim in enumerate(shape):
            if not isinstance(dim, int):
                raise TypeError(f"""All dimensions of shape must be integers. (Got illegal {type(dim).__name__} ({dim}) at index {i})""")
        
        self.x = shape[0]
        self.y = shape[1]
        
        self.bitmap = bitmap or np.zeros((self.x, self.y), dtype=int)
        
    def __setitem__(self, key, value):
        self.bitmap[key] = value
    def __getitem__(self, key):
        return self.bitmap[key]
        
    def __str__(self) -> str:
        s = ''
        
        for x in self.bitmap:
            for y in x:
                s += ' XO'[y]
            s += '|\n'
        s += '\n'
        return s
    
    def dist_from_centre(self, coord: tuple[int, int]) -> int:
        """Obtain the Manhattan distance from the centre of the bitmap to the given coordinates."""
        
        midx = self.x // 2
        midy = self.y // 2
        x, y = coord
        
        return abs(x - midx) + abs(y - midy)
    
    def get_random_start_point(self, dev_from_middle: float, middle_variation: float) -> tuple[int, int]:
        """Function to get a random point to start the random patch.
        First determines a random distance based on dev_from_middle and middle_variation, then selects a random coordinate with the equivalent Manhatten distance."""
        
        chance_to_stop = (1 - dev_from_middle) ** 2.25
        averaging_level = round((1 - middle_variation) * 10)
        
        dist = 0
        for _ in range(averaging_level):
            for _ in range(self.x // 2 + self.y // 2):
                if rn.random() < chance_to_stop:
                    break
                dist += 1
        dist = round(dist / averaging_level)
        
        dist = min(dist, self.x // 2 + self.y // 2)
        
        possible_starts = [i for i in product(range(self.x), range(self.y)) if self.dist_from_centre(i) == dist]
        return rn.choice(possible_starts)

    def recur_spread(self, x: int, y: int, dist: int, density: float) -> None:
        """Nested procedure to recursively 'spread' ore in any of the 8 cardinal directions.
        Args:
            x (int): The current x coordinate.
            y (int): The current y coordinate.
            dist (int): The remaining distance.
            density (float): Refer to .random.
        """
        
        if dist == 0:
            return
        
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        
        # find a direction that doesn't go out of range
        rx, ry = rn.choice(dirs)
        while x + rx not in range(self.x - 1) or y + ry not in range(self.y - 1):
            # 75% chance to skip squares with ore
            if self[x + rx, y + ry] == 0 or rn.random() < 0.75:
                rx, ry = rn.choice(dirs)
        
        if density + 0.1 > rn.random():
            self[x + rx, y + ry] = 1
        
        self.recur_spread(x + rx, y + ry, dist - 1, density)
    
    def random(self,*, dev_from_middle: float=0.5, middle_variation: float=0.25, density: float=0.5) -> 'Ore':
        """Generate and return a random Ore bitmap.
        Args:
            dev_from_middle (float): 
                A float between 0 and 1 determining how far the start point is from the centre of the bitmap. 
                The lower, the closer to the middle.
            middle_variation (float):
                A float between 0 and 1 determining how much the middle point varies when the same deviation from middle is given.
                Takes an average of middle_variation * 10 values.
            density (float): 
                A floating point number between 0 and 1 to define how dense the ore patch is.
        Returns:
            Ore: The randomly generated ore patch stored in a class.
        """
        
        if not 0 <= dev_from_middle <= 1:
                raise ValueError(f"""Deviation from middle must be between 0 and 1. (Got {dev_from_middle})""")
        if not 0 <= middle_variation <= 1:
            raise ValueError(f"""Middle variation must be between 0 and 1. (Got {middle_variation})""")
        
        random_ore = Ore((self.x, self.y))
        
        x, y = random_ore.get_random_start_point(dev_from_middle, middle_variation)
        
        for _ in range(round(density * 10)):
            random_ore.recur_spread(x, y, rn.randint(self.x // 4 + self.y // 4, self.x // 2 + self.y // 2), density)
        random_ore[x, y] = 1
        
        return random_ore
