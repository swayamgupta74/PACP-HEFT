import math #for math.floor
class Task:
    def __init__(self, id, size):
        self.id = id
        self.size = size  # Task runtime in seconds (to be scaled by VM capacity)
        self.predecessors = []
        self.successors = []
class VM:
    def __init__(self, type_id, processing_capacity, cost_per_interval,efficiency=None):
        self.type_id = type_id
        self.processing_capacity = processing_capacity  # Tasks per second
        self.cost_per_interval = cost_per_interval  # Cost per billing interval
        self.efficiency = efficiency  # Efficiency rate (rho / cost)
        self.tasks = []  # Tasks assigned to this VM
        self.lease_start_time = 0
        self.lease_finish_time = 0

def parse_dag_file(dag_file_path):
    tasks = {}
    edges = {}
    file_sizes = {}

    # Read the DAG file
    with open(dag_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse files
    for line in lines:
        if line.startswith('FILE'):
            parts = line.strip().split()
            file_name = parts[1]
            file_size = int(parts[2]) / (1024 * 1024)  # Convert bytes to MB
            file_size=round(file_size,2)
            file_sizes[file_name] = file_size

    # Parse tasks
    for line in lines:
        if line.startswith('TASK'):
            parts = line.strip().split()
            task_id = parts[1]
            runtime = float(parts[3])  # Runtime in seconds
            tasks[task_id] = Task(task_id, size=runtime)
    
    # Parse inputs/outputs
    task_inputs = {}
    task_outputs = {}
    for line in lines:
        if line.startswith('INPUTS'):
            parts = line.strip().split()
            task_id = parts[1]
            input_files = parts[2:] if len(parts) > 2 else []
            task_inputs[task_id] = input_files
        if line.startswith('OUTPUTS'):
            parts = line.strip().split()
            task_id = parts[1]
            output_files = parts[2:] if len(parts) > 2 else []
            task_outputs[task_id] = output_files
    
    # Parse edges
    for line in lines:
        if line.startswith('EDGE'):
            parts = line.strip().split()
            parent_id = parts[1]
            child_id = parts[2]
            if parent_id in tasks and child_id in tasks:
                if child_id not in edges:
                    edges[child_id] = {}
                # Data size is sum of parent's output files (output files can be multiple)
                data_size = sum(file_sizes.get(f, 0) for f in task_outputs.get(parent_id, [])) 
                edges[child_id][parent_id] = data_size
                tasks[child_id].predecessors.append(parent_id)
                tasks[parent_id].successors.append(child_id)
    
    # Add entry and exit tasks
    entry_tasks = [tid for tid in tasks if not tasks[tid].predecessors]
    exit_tasks = [tid for tid in tasks if not tasks[tid].successors]
    
    if len(entry_tasks) > 1:
        tasks['t_en'] = Task('t_en', size=0)
        for et in entry_tasks:
            edges[et] = {'t_en': 0}
            tasks[et].predecessors.append('t_en')
            tasks['t_en'].successors.append(et)
    
    if len(exit_tasks) > 1:
        tasks['t_ex'] = Task('t_ex', size=0)
        for et in exit_tasks:
            if 't_ex' not in edges:
                edges['t_ex'] = {}
            edges['t_ex'][et] = 0
            tasks['t_ex'].predecessors.append(et)
            tasks[et].successors.append('t_ex')
    
    return tasks, edges
    
def InitializePriority(tasks, edges, resource_pool, bandwidth):
    avg_processing_capacity = sum(vm.processing_capacity for vm in resource_pool) / len(resource_pool)
    avg_comm_cost = 1 / bandwidth  # Time per MB
    
    ranku = {}
    def calc_ranku(task_id):
        if task_id in ranku:
            return ranku[task_id]
        
        task = tasks[task_id]
        # Average computation cost
        w_i = task.size / avg_processing_capacity
        
        # Successor communication and rank
        max_succ_cost = 0
        for succ_id in task.successors:
            comm_cost = edges.get(succ_id, {}).get(task_id, 0) * avg_comm_cost
            succ_rank = calc_ranku(succ_id)
            max_succ_cost = max(max_succ_cost, comm_cost + succ_rank)
        
        ranku[task_id] = w_i + max_succ_cost

        return ranku[task_id]
    
    for task_id in tasks:
        calc_ranku(task_id)
   
   # Filter out metadata keys and sort by value (descending), task_id (ascending for ties)
    tasks = [(tid, val) for tid, val in ranku.items() if tid not in ['t_ex', 't_en']]
    tasks.sort(key=lambda x: (-x[1], x[0]))

    # updating ranku dictionary with task IDs and unique ranks starting from 0
    ranku = {tid: i for i, (tid, _) in enumerate(tasks)}
    
    return ranku
    
def calculate_efficiency(vm):
    return vm.processing_capacity / vm.cost_per_interval
def rtl(resource_pool):
    # Calculate efficiency for each VM type
    for vm_type in resource_pool:
        vm_type.efficiency = calculate_efficiency(vm_type)
    
    # Sort VM types by efficiency (descending)
    RTL = sorted(resource_pool, key=lambda x: x.efficiency, reverse=True)
    return RTL
    
    
# Algorithm 2: LeaseVM - Lease VMs based on budget and efficiency
def LeaseVM(resource_pool, budget):
    time_interval = 60  # Billing interval in minutes
    M = {}  # Dictionary to store {type_id: leased_intervals}
    
    remaining_budget = budget
    RTL = rtl(resource_pool)
    
    while RTL:
        # Select VM type with highest efficiency
        vm_type = RTL[0]
        intervals = math.floor(remaining_budget / vm_type.cost_per_interval)

        # Record leased intervals for this VM type
        if intervals > 0:  # Only add if intervals are leased
            M[vm_type.type_id] = M.get(vm_type.type_id, 0) + intervals

        # Update remaining budget
        remaining_budget -= intervals * vm_type.cost_per_interval

        # Filter VM types that can be afforded
        RTL = [vm_type for vm_type in RTL if vm_type.cost_per_interval <= remaining_budget]

    return M
    

def BuildSolution(Tasks, Edges, M, P0):
    """
    Implements Algorithm 3: BuildSolution from PACP-HEFT paper.
    Args:
        Tasks: Dictionary of task_id -> Task object
        Edges: Dictionary of task_id -> {parent_id: data_size}
        M: Dictionary of VM type_id -> number of leased intervals
        P: Dictionary of task_id -> priority rank (lower value = higher priority)

    Returns:
        S: Tuple (M, AL) where AL is list of (task_id, vm, ST, FT)
    """
    # Step 1: Sort all tasks according to priority P0
    sorted_tasks = sorted(P0.items(), key=lambda x: x[1])  # ascending order
    sorted_task_ids = [tid for tid, _ in sorted_tasks]

    AL = []  # Assignments: (task_id, vm, ST, FT)
    vm_instances = {}  # vm_type -> list of VM instances (each is a dict with its own task schedule)

    # Initialize VM instances
    for vm_type, count in M.items():
        vm_instances[vm_type] = []
        for i in range(count):
            vm_instances[vm_type].append({
                "tasks": [],  # scheduled tasks
                "MST": 0,     # lease start
                "MFT": 0      # lease finish
            })

    def compute_EST(task_id, vm):
        preds = Tasks[task_id].predecessors
        if not preds:
            return vm['MFT']  # No parent; can start anytime
        max_parent_ft = 0
        for parent in preds:
            parent_ft = next((x[3] for x in AL if x[0] == parent), 0)
            comm_cost = Edges.get(task_id, {}).get(parent, 0) / bandwidth
            max_parent_ft = max(max_parent_ft, parent_ft + comm_cost)
        return max(max_parent_ft, vm['MFT'])

    # Step 2: Loop over each task
    for task_id in sorted_task_ids:
        task = Tasks[task_id]
        best_vm = None
        best_ft = float('inf')
        best_instance = None
        no_extra_cost_vms = []

        for vm_type in M:
            for vm in vm_instances[vm_type]:
                est = compute_EST(task_id, vm)
                et = task.size / [v.processing_capacity for v in resource_pool if v.type_id == vm_type][0]
                ft = est + et
                duration = ft - vm['MST']
                intervals_needed = math.ceil(duration / 60)
                vm_leased_intervals = M[vm_type]

                if intervals_needed <= vm_leased_intervals:
                    no_extra_cost_vms.append((ft, vm, vm_type))

                # Track VM with min FT
                if ft < best_ft:
                    best_ft = ft
                    best_vm = vm
                    best_instance = vm_type

        # Select the best VM
        if no_extra_cost_vms:
            # Choose VM with minimum FT among no-extra-cost VMs
            selected_ft, selected_vm, selected_type = min(no_extra_cost_vms, key=lambda x: x[0])
        else:
            selected_ft = best_ft
            selected_vm = best_vm
            selected_type = best_instance

        # Assign task
        et = task.size / [v.processing_capacity for v in resource_pool if v.type_id == selected_type][0]
        st = selected_ft - et
        selected_vm['tasks'].append((task_id, st, selected_ft))
        selected_vm['MFT'] = max(selected_vm['MFT'], selected_ft)
        selected_vm['MST'] = min(selected_vm['MST'], st) if selected_vm['tasks'] else st

        AL.append((task_id, selected_type, st, selected_ft))

    return (M, AL)
    
    
def Priority1(T,E,S0):
    return 0
def Priority2(T,E,S1):
    return 0
def CriticalTaskOptimizer(S2):
    return 0  
    
# def best():
#     # Comparison Logic
# #Both Feasible: If $ WSC_s1 \leq B $ and $ WSC_s2 \leq B $, return the solution with the lower $ WST $.
# #Both Infeasible: If $ WSC_s1 > B $ and $ WSC_s2 > B $, return the solution with the lower $ WSC $.
# #One Feasible: If only one solution is feasible, return the feasible one (the infeasible one is discarded).

#     # Determine the best solution
#     s1_feasible = wsc_s1 <= budget
#     s2_feasible = wsc_s2 <= budget

#         # Determine the best solution
#     s1_feasible = wsc_s1 <= budget
#     s2_feasible = wsc_s2 <= budget

#     if s1_feasible and s2_feasible:
#         return (tasks_s1, vms_s1) if wst_s1 <= wst_s2 else (tasks_s2, vms_s2)
#     elif not s1_feasible and not s2_feasible:
#         return (tasks_s1, vms_s1) if wsc_s1 < wsc_s2 else (tasks_s2, vms_s2)
#     elif not s1_feasible:
#         return (tasks_s2, vms_s2)  # S2 is feasible, so it's better
#     else:  # s2_feasible is False
#         return (tasks_s1, vms_s1)  # S1 is feasible, so it's better

#     return 0    

def PACP_HEFT(T,E,R,B):
    RTL=rtl(R)
    S={}
    
    M=LeaseVM(RTL,B)
    P0=InitializePriority(T,E,R,B)
    S0=BuildSolution(T,E,M,P0)  
    print(S0)


# def PACP_HEFT(T,E,R,B):
#     RTL=rtl(R)
#     S={}
#     while RTL :
#         M=LeaseVM(RTL,B)
#         P0=InitializePriority(T,E,R,B)
#         S0=BuildSolution(T,E,M,P0)
#         better=True
#         while better:
#             P1=Priority1(T,E,S0)
#             S1=best(S0,BuildSolution(P1))
#             if S1 better than S0 :
#                 S0=S1
#                 continue
#             P2=Priority2(T,E,S1)
#             S2=best(S1,BuildSolution(P2))
#             if S2 better than S0:
#                 S0=S2
#                 continue
#             S3=best(S2,CriticalTaskOptimizer(S2))
#             if S3 better than S0:
#                 S0=S3
#                 continue
#             better=False
#         S=best(S0,S)
#         RTL=RTL[1:]
        
#     return S
    


if __name__ == "__main__":
    dag_file_path = "CYBERSHAKE.n.50.0.dag"  # Ensure this file exists in the directory
    budget = 9  # Adjusted for CyberShake
    #time_interval = 60  # Billing interval in minutes, here constant so defined in LeaseVM function
    bandwidth = 20  # MB/s
    resource_pool = [
        #VM(0, processing_capacity=8, cost_per_interval=4),   # Small instance
        VM("m1.small", processing_capacity=1, cost_per_interval=0.044),    # 1 unit, $0.044/hour
        VM("m1.medium", processing_capacity=2, cost_per_interval=0.087),   # 2 units, $0.087/hour
        VM("m3.medium", processing_capacity=3, cost_per_interval=0.067),   # 3 units, $0.067/hour
        VM("m1.large", processing_capacity=4, cost_per_interval=0.175),    # 4 units, $0.175/hour
        VM("m3.large", processing_capacity=6.5, cost_per_interval=0.133),  # 6.5 units, $0.133/hour
        VM("m1.xlarge", processing_capacity=8, cost_per_interval=0.350),   # 8 units, $0.350/hour
        VM("m3.xlarge", processing_capacity=13, cost_per_interval=0.266),  # 13 units, $0.266/hour
        VM("m3.2xlarge", processing_capacity=26, cost_per_interval=0.532) # 26 units, $0.532/hour
    ]
    tasks, edges = parse_dag_file(dag_file_path)
    
#    edges = {
#     "t2": {"t1": 1.0},
#     "t3": {"t2": 2.0}
#     }
#     tasks = {
#     "t1": Task(id="t1", size=10.0, predecessors=[], successors=["t2"]),
#     "t2": Task(id="t2", size=15.0, predecessors=["t1"], successors=["t3"]),
#     "t3": Task(id="t3", size=20.0, predecessors=["t2"], successors=[])
#     }  
    
    PACP_HEFT(tasks, edges,resource_pool,budget)
    
    #print(parse_dag_file(dag_file_path))
    #print(InitializePriority(tasks, edges, resource_pool, bandwidth))
    #M=LeaseVM(resource_pool, budget, time_interval)
    #print(M)
    
    
    
    
    
    
    
