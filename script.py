import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# ======================================
# CONFIGURATION PARAMETERS
# ======================================
N_GEN_OPTIMIZATIONS = 1000
NUM_DRONES = 10
AREA_SIZE = 1000
MAX_DEPTH = 800
MOVEMENT_RANGE = 500
SURFACE_BUOY = np.array([AREA_SIZE/2, AREA_SIZE/2, 0])

LINK_PROFILES = {
    'radio': {'range': 20, 'color': 'red', 'priority': 3},
    'optical': {'range': 500, 'color': 'blue', 'priority': 2},
    'acoustic': {'range': 1000, 'color': 'green', 'priority': 1}
}

# ======================================
# INITIAL POSITIONS AND BOUNDS
# ======================================
np.random.seed(42)
initial_positions = np.column_stack([
    np.random.rand(NUM_DRONES) * AREA_SIZE,
    np.random.rand(NUM_DRONES) * AREA_SIZE,
    np.random.rand(NUM_DRONES) * MAX_DEPTH
])

# Force initial drone near buoy, try promote at least one EM based link
initial_positions[0] = SURFACE_BUOY + np.array([5, -5, 5])

# Create movement bounds
movement_bounds = []
for pos in initial_positions:
    movement_bounds.extend([
        (pos[0]-MOVEMENT_RANGE, pos[0]+MOVEMENT_RANGE),
        (pos[1]-MOVEMENT_RANGE, pos[1]+MOVEMENT_RANGE),
        (pos[2]-MOVEMENT_RANGE, pos[2]+MOVEMENT_RANGE)
    ])

# ======================================
# COMMUNICATION FUNCTIONS
# ======================================
def get_link_type(distance):
    if distance <= LINK_PROFILES['radio']['range']:
        return 'radio'
    elif distance <= LINK_PROFILES['optical']['range']:
        return 'optical'
    elif distance <= LINK_PROFILES['acoustic']['range']:
        return 'acoustic'
    return None

def create_communication_network(positions):
    G = nx.Graph()
    positions = np.vstack([positions, SURFACE_BUOY])
    for i in range(len(positions)):
        G.add_node(i, pos=positions[i])
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            link_type = get_link_type(dist)
            if link_type:
                current_priority = LINK_PROFILES[link_type]['priority']
                existing = G.get_edge_data(i, j)
                existing_priority = existing['priority'] if existing else -1
                if current_priority > existing_priority:
                    G.add_edge(i, j,
                               link_type=link_type,
                               distance=dist,
                               priority=current_priority)
    return G

def find_optimal_path(G, target_idx):
    fast_links = ['radio', 'optical']
    H_fast = G.edge_subgraph([
        (u, v) for u, v, d in G.edges(data=True)
        if d['link_type'] in fast_links
    ]).copy()

    surface_node = len(G.nodes) - 1
    try:
        return nx.shortest_path(H_fast, source=target_idx, target=surface_node,
                                weight='distance')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

# ======================================
# OPTIMIZATION PROBLEM
# ======================================
class DroneOptimizationProblem(ElementwiseProblem):
    def __init__(self, movement_bounds, fixed_idx, fixed_pos):
        self.fixed_idx = fixed_idx
        self.fixed_pos = fixed_pos
        self.n_drones = int(len(movement_bounds) / 3)
        xl = [b[0] for b in movement_bounds]
        xu = [b[1] for b in movement_bounds]
        super().__init__(n_var=len(xl), n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.objective(x)

    def objective(self, x):
        positions = x.reshape(-1, 3)

        # Fix the target drone's position
        positions[self.fixed_idx] = self.fixed_pos

        G = create_communication_network(positions)
        path = find_optimal_path(G, self.fixed_idx)
        if not path:
            return 1e9  # Penalty for no connection

        acoustic_penalty = 0
        total_distance = 0
        for i in range(len(path)-1):
            edge_data = G.edges[path[i], path[i+1]]
            total_distance += edge_data['distance']
            if edge_data['link_type'] == 'acoustic':
                acoustic_penalty += 10000

        return (total_distance * 0.1) + (len(path) * 1000) + acoustic_penalty

# Identify fixed drone
fixed_idx = np.argmax(initial_positions[:, 2])
fixed_pos = initial_positions[fixed_idx].copy()

problem = DroneOptimizationProblem(movement_bounds, fixed_idx, fixed_pos)

# ======================================
# OPTIMIZATION EXECUTION
# ======================================
algorithm = GA(
    pop_size=100,
    eliminate_duplicates=True,
    mutation=PM(prob=0.2, eta=15),
    crossover=SBX(prob=0.9, eta=20),
    sampling=np.vstack([
        initial_positions.flatten(),
        np.random.uniform(low=[b[0] for b in movement_bounds],
                          high=[b[1] for b in movement_bounds],
                          size=(99, len(movement_bounds)))
    ]),
    n_offsprings=50
)

termination = get_termination("n_gen", N_GEN_OPTIMIZATIONS)
res = pymoo_minimize(
    problem,
    algorithm,
    termination,
    seed=42,
    verbose=True,
    save_history=True
)

# ======================================
# VISUALIZATION
# ======================================
def plot_network(positions, title, ax):
    ax.clear()
    xx, yy = np.meshgrid([0, AREA_SIZE], [0, AREA_SIZE])
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='blue')
    ax.scatter(*SURFACE_BUOY, s=200, marker='*', c='gold', edgecolor='k')

    all_positions = np.vstack([positions, SURFACE_BUOY])
    G = create_communication_network(positions)
    depths = [pos[2] for pos in positions]
    target_idx = np.argmax(depths)
    path = find_optimal_path(G, target_idx)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               s=60, c='gray', edgecolors='k', alpha=0.7)
    ax.scatter(positions[target_idx, 0], positions[target_idx, 1], positions[target_idx, 2],
               s=100, c='red', edgecolors='k')

    for u, v, d in G.edges(data=True):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        ax.plot([pos_u[0], pos_v[0]],
                [pos_u[1], pos_v[1]],
                [pos_u[2], pos_v[2]],
                color=LINK_PROFILES[d['link_type']]['color'],
                alpha=0.4,
                linewidth=2)

    description = f"No path found"  # Default message
    if path:
        total_distance = 0.0
        link_types = []
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)
            color = LINK_PROFILES[edge_data['link_type']]['color']
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            ax.plot([pos_u[0], pos_v[0]],
                    [pos_u[1], pos_v[1]],
                    [pos_u[2], pos_v[2]],
                    color=color,
                    linewidth=3.5,
                    alpha=0.9)
            total_distance += edge_data['distance']
            link_types.append(edge_data['link_type'])

        hops_str = ", ".join(link_types)
        description = f"{title} Path: {total_distance:.1f}m over {len(link_types)} hops ({hops_str})"
        
    ax.set_title(title)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Depth (m)')
    ax.invert_zaxis()
    ax.grid(True)
    ax.view_init(elev=25, azim=-45)
    

    
    return description  # Return the generated description

# Create figure and add text labels
# Create figure and add text labels
fig = plt.figure(figsize=(16, 10)) 
plt.subplots_adjust(bottom=0.25)  

ax1 = fig.add_subplot(121, projection='3d')
initial_desc = plot_network(initial_positions, "Initial Configuration", ax1)

ax2 = fig.add_subplot(122, projection='3d')
optimized_desc = plot_network(res.X.reshape(-1, 3), "GA-Optimized Configuration", ax2)

# Add path labels under subplots
fig.text(0.25, 0.12, initial_desc, ha='center', va='bottom', fontsize=10)
fig.text(0.75, 0.12, optimized_desc, ha='center', va='bottom', fontsize=10)

# Add configuration parameters box
config_str = (
    "Configuration Parameters:\n"
    f"N_GEN_OPTIMIZATIONS = {N_GEN_OPTIMIZATIONS}\n"
    f"NUM_DRONES = {NUM_DRONES}\n"
    f"AREA_SIZE = {AREA_SIZE}\n" 
    f"MAX_DEPTH = {MAX_DEPTH}\n"
    f"MOVEMENT_RANGE = {MOVEMENT_RANGE}"
)

fig.text(0.5, 0.05, config_str, 
         ha='center', va='bottom', 
         fontsize=10, linespacing=1.8,
         bbox=dict(facecolor='white', alpha=0.8, 
                   edgecolor='lightgray', boxstyle='round,pad=0.5'))

plt.show()