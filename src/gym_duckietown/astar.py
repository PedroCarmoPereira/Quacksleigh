import heapq

def _heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(env, start, goal):
    width, height = env.grid_width, env.grid_height
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: _heuristic(start, goal)}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
            
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                tile = env._get_tile(neighbor[0], neighbor[1])
                if tile is None or not tile['drivable']:
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + _heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return []