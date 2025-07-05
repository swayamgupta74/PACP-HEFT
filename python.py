import math

class Task:
    def __init__(self, id, size):
        self.id = id
        self.size = size  # Task runtime in seconds
        self.predecessors = []
        self.successors = []

class VM:
    def __init__(self, type_id, processing_capacity, cost_per_interval):
        self.type_id = type_id
        self.processing_capacity = processing_capacity  # Tasks per second
        self.cost_per_interval = cost_per_interval  # Cost per hour (as per original pricing)
        self.tasks = []  # Tasks assigned to this VM
        self.lease_start_time = 0
        self.lease_finish_time = 0

def parse_dag_file(dag_file_path):
    """
    Parse DAG file to create tasks and edges dictionaries.
    Args:
        dag_file_path: Path to the DAG file.
    Returns:
        Tuple (tasks, edges) where tasks is a dictionary of Task objects and edges is a dictionary of edge weights.
    """
    tasks = {}
    edges = {}
    file_sizes = {}

    with open(dag_file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('FILE'):
            parts = line.strip().split()
            file_name = parts[1]
            file_size = int(parts[2]) / (1024 * 1024)  # Convert bytes to MB
            file_sizes[file_name] = round(file_size, 2)

    for line in lines:
        if line.startswith('TASK'):
            parts = line.strip().split()
            task_id = parts[1]
            runtime = float(parts[3])  # Runtime in seconds
            tasks[task_id] = Task(task_id, size=runtime)
    
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
    
    for line in lines:
        if line.startswith('EDGE'):
            parts = line.strip().split()
            parent_id = parts[1]
            child_id = parts[2]
            if parent_id in tasks and child_id in tasks:
                if child_id not in edges:
                    edges[child_id] = {}
                # Use specific output files relevant to the edge (simplified assumption)
                data_size = next((file_sizes.get(f, 0) for f in task_outputs.get(parent_id, []) if f in task_inputs.get(child_id, [])), 0)
                edges[child_id][parent_id] = data_size
                tasks[child_id].predecessors.append(parent_id)
                tasks[parent_id].successors.append(child_id)
    
    entry_tasks = [tid for tid in tasks if not tasks[tid].predecessors]
    exit_tasks = [tid for tid in tasks if not tasks[tid].successors]
    
    if len(entry_tasks) > 1:
        tasks['t_en'] = Task('t_en', size=0)
        for et in entry_tasks:
            if 't_en' not in edges:
                edges['t_en'] = {}
            edges[et] = {'t_en': 0}  # Zero weight for entry edges
            tasks[et].predecessors.append('t_en')
            tasks['t_en'].successors.append(et)
    
    if len(exit_tasks) > 1:
        tasks['t_ex'] = Task('t_ex', size=0)
        for et in exit_tasks:
            if 't_ex' not in edges:
                edges['t_ex'] = {}
            edges['t_ex'][et] = 0  # Zero weight for exit edges
            tasks['t_ex'].predecessors.append(et)
            tasks[et].successors.append('t_ex')
    
    return tasks, edges

# Replace duplicate rtl with the filtered version
def rtl(resource_pool):
    """
    Rank VM types by efficiency and filter dominated types.
    Args:
        resource_pool: List of VM objects.
    Returns:
        Sorted list of VM types with dominated types removed.
    """
    for vm_type in resource_pool:
        vm_type.efficiency = calculate_efficiency(vm_type)
    filtered_pool = []
    for i, vm1 in enumerate(resource_pool):
        is_dominated = False
        for vm2 in resource_pool:
            if vm1 != vm2 and vm2.processing_capacity >= vm1.processing_capacity and vm2.cost_per_interval <= vm1.cost_per_interval and (vm2.processing_capacity > vm1.processing_capacity or vm2.cost_per_interval < vm1.cost_per_interval):
                is_dominated = True
                break
        if not is_dominated:
            filtered_pool.append(vm1)
    return sorted(filtered_pool, key=lambda x: x.efficiency, reverse=True)

def InitializePriority(tasks, edges, resource_pool, bandwidth):
    """
    Compute initial priorities using upward rank.
    Args:
        tasks: Dictionary of Task objects (filtered to exclude t_en and t_ex).
        edges: Dictionary of edge weights (data sizes in MB).
        resource_pool: List of VM types.
        bandwidth: Communication bandwidth in MB/s.
    Returns:
        Dictionary of task IDs to priority ranks.
    """
    avg_processing_capacity = sum(vm.processing_capacity for vm in resource_pool) / len(resource_pool)
    avg_comm_cost = 1 / bandwidth  # Time per MB in seconds
    
    ranku = {}
    def calc_ranku(task_id):
        if task_id in ranku or task_id in ['t_en', 't_ex']:  # Skip artificial tasks
            return ranku.get(task_id, 0)
        
        if task_id not in tasks:
            return 0  # Handle cases where task_id is not in filtered tasks
        
        task = tasks[task_id]
        w_i = task.size / avg_processing_capacity
        
        max_succ_cost = 0
        for succ_id in task.successors:
            comm_cost = edges.get(succ_id, {}).get(task_id, 0) * avg_comm_cost
            succ_rank = calc_ranku(succ_id)
            max_succ_cost = max(max_succ_cost, comm_cost + succ_rank)
        
        ranku[task_id] = w_i + max_succ_cost
        return ranku[task_id]
    
    for task_id in tasks:
        calc_ranku(task_id)
   
    tasks_list = [(tid, val) for tid, val in ranku.items() if tid not in ['t_en', 't_ex']]
    tasks_list.sort(key=lambda x: (-x[1], x[0]))
    return {tid: i for i, (tid, _) in enumerate(tasks_list)}

def calculate_efficiency(vm):
    return vm.processing_capacity / vm.cost_per_interval

def rtl(resource_pool):
    for vm_type in resource_pool:
        vm_type.efficiency = calculate_efficiency(vm_type)
    return sorted(resource_pool, key=lambda x: x.efficiency, reverse=True)

def LeaseVM(RTL, budget, T=3600):  # T = 1 hour in seconds
    """
    Lease VMs within budget.
    Args:
        RTL: Ranked list of VM types.
        budget: Budget in dollars per hour.
        T: Billing interval in seconds (default 3600).
    Returns:
        Set of (vm_type, instance_id) tuples.
    """
    M = set()  # Set of (vm_type, instance_id) tuples
    remaining_budget = budget
    instance_id = 0
    
    while RTL and remaining_budget > 0:
        vm_type = RTL[0]
        cost_per_hour = vm_type.cost_per_interval
        max_instances = math.floor(remaining_budget / cost_per_hour)
        if max_instances > 0:
            for _ in range(min(max_instances, 5)):  # Limit to 5 instances per type
                M.add((vm_type.type_id, instance_id))
                remaining_budget -= cost_per_hour
                instance_id += 1
        RTL = RTL[1:]  # Move to next VM type
        
    return M

def BuildSolution(Tasks, Edges, M, P0, resource_pool, bandwidth=20, T=3600):
    """
    Build a scheduling solution using insertion-based scheduling on leased VMs.
    Args:
        Tasks: Dictionary of Task objects.
        Edges: Dictionary of edge weights (data sizes in MB).
        M: List of (vm_type, instance_id) tuples representing leased VMs.
        P0: Dictionary of task priorities (task_id -> rank).
        resource_pool: List of VM types.
        bandwidth: Communication bandwidth in MB/s.
        T: Billing interval in seconds.
    Returns:
        Tuple (M, AL) where AL is the assignment list [(task_id, (vm_type, instance_id), ST, FT)].
    """
    AL = []  # Assignment list: (task_id, (vm_type, instance_id), ST, FT)
    vm_schedules = {vm_id: [] for vm_id in M}  # (vm_type, instance_id) -> list of (ST, FT)

    def compute_EFT(task_id, vm_type, instance_id):
        task = Tasks[task_id]  # Retrieve Task object
        vm_id = (vm_type, instance_id)
        est = 0
        preds = task.predecessors
        if preds:
            max_parent_ft = 0
            for parent in preds:
                parent_ft = next((ft for t_id, _, _, ft in AL if t_id == parent), 0)
                comm_cost = (Edges.get(task_id, {}).get(parent, 0) / bandwidth 
                            if vm_type != next((v[0] for t_id, v, _, _ in AL if t_id == parent), vm_type) 
                            else 0)
                max_parent_ft = max(max_parent_ft, parent_ft + comm_cost)
            est = max_parent_ft

        schedule = vm_schedules[vm_id]
        slots = [(0, est)] + sorted(schedule, key=lambda x: x[0]) + [(float('inf'), float('inf'))]
        min_eft = float('inf')
        best_st = est
        et = task.size / next(v.processing_capacity for v in resource_pool if v.type_id == vm_type)  # Use task object

        for i in range(len(slots) - 1):
            st = max(slots[i][1], est)
            ft = st + et
            if ft <= slots[i + 1][0] and ft < min_eft:
                min_eft = ft
                best_st = st

        return best_st, min_eft

    sorted_tasks = sorted(P0.items(), key=lambda x: x[1], reverse=True)
    sorted_task_ids = [tid for tid, _ in sorted_tasks]

    for task_id in sorted_task_ids:
        best_eft = float('inf')
        best_vm = None
        best_st = 0

        for vm_type, instance_id in M:
            st, eft = compute_EFT(task_id, vm_type, instance_id)
            if eft < best_eft:
                best_eft = eft
                best_vm = (vm_type, instance_id)
                best_st = st

        if best_vm:
            vm_type, instance_id = best_vm
            et = Tasks[task_id].size / next(v.processing_capacity for v in resource_pool if v.type_id == vm_type)  # Use Tasks dict directly
            ft = best_st + et
            vm_id = (vm_type, instance_id)
            AL.append((task_id, vm_id, round(best_st, 2), round(ft, 2)))
            vm_schedules[vm_id].append((best_st, ft))

    return (M, AL)

# Update compute_thetaD to remove premature VM check
def compute_thetaD(tasks, edges, schedule, bandwidth):
    task_vm = {tid: vm[0] for tid, (vm, _), _, _ in schedule}
    task_et = {tid: ft - st for tid, _, st, ft in schedule}
    successors = {tid: [] for tid in tasks}
    for child, parent_map in edges.items():
        for parent in parent_map:
            if parent in tasks:
                successors[parent].append(child)

    thetaD = {}
    def dfs(tid):
        if tid in thetaD or tid not in tasks:
            return thetaD.get(tid, 0)
        if not successors.get(tid, []):
            thetaD[tid] = task_et.get(tid, 0)
            return thetaD[tid]
        max_path = 0
        for succ in successors[tid]:
            comm = edges.get(succ, {}).get(tid, 0)
            vm_succ = task_vm.get(succ, None)
            vm_curr = task_vm.get(tid, None)
            comm_cost = comm / bandwidth if vm_succ != vm_curr and vm_succ is not None and vm_curr is not None else 0
            succ_theta = dfs(succ)
            max_path = max(max_path, comm_cost + succ_theta + task_et.get(tid, 0))
        thetaD[tid] = max_path
        return thetaD[tid]

    for tid in tasks:
        dfs(tid)
    filtered = [(tid, val) for tid, val in thetaD.items() if tid not in ['t_en', 't_ex']]
    return {tid: i for i, (tid, _) in enumerate(sorted(filtered, key=lambda x: -x[1]))}

def compute_thetaT(AL):
    WST = max(ft for _, _, _, ft in AL)
    thetaT = {tid: WST - st for tid, _, st, _ in AL if tid not in ['t_en', 't_ex']}
    return {tid: i for i, (tid, _) in enumerate(sorted(thetaT.items(), key=lambda x: -x[1]))}

def compute_slack_refined(AL, edges, bandwidth, vm_speeds):
    """
    Compute refined slack times for tasks based on reverse scheduling.
    Args:
        AL: List of (task_id, (vm_type, instance_id), ST, FT) tuples.
        edges: Dictionary of edge weights (data sizes in MB).
        bandwidth: Communication bandwidth in MB/s.
        vm_speeds: Dictionary of vm_type to processing capacity.
    Returns:
        Tuple (deltaT, reverse_st) where deltaT is slack times and reverse_st is reverse start times.
    """
    # Create task_map with (tid, idx) mapping to (vm, st, ft)
    task_map = {(tid, idx): (vm, st, ft) for tid, (vm, idx), st, ft in AL}
    WST = max(ft for _, _, _, ft in AL)
    reverse_st = {}
    reverse_ft = {tid: WST for tid in {t for t, _, _, _ in AL}}  # Unique task IDs

    tasks_sorted = sorted(AL, key=lambda x: -x[3])  # Sort by FT descending
    # Initialize fpt with WST for each unique (vm, idx)
    fpt = {(vm, idx): WST for _, (vm, idx), _, _ in AL}

    succ_map = {tid: [] for tid in {t for t, _, _, _ in AL}}
    for child, parent_map in edges.items():
        for parent in parent_map:
            if parent in succ_map:
                succ_map[parent].append(child)

    for tid, (vm, idx), _, _ in tasks_sorted:
        succs = succ_map.get(tid, [])
        min_succ_start = float('inf')
        for succ in succs:
            if succ not in {t for t, _, _, _ in AL}:
                continue
            succ_vm, succ_idx, succ_st, _ = next((v, i, st, ft) for t, (v, i), st, ft in AL if t == succ)
            tt = edges.get(succ, {}).get(tid, 0) / bandwidth if vm != succ_vm else 0
            min_succ_start = min(min_succ_start, reverse_st.get(succ, WST) - tt)
        min_succ_start = min(min_succ_start, fpt[(vm, idx)])
        et = task_map[(tid, idx)][2] - task_map[(tid, idx)][1]  # ft - st
        reverse_ft[tid] = min_succ_start
        reverse_st[tid] = min_succ_start - et
        fpt[(vm, idx)] = reverse_st[tid]
    
    # Calculate deltaT using the correct (tid, idx) from task_map
    deltaT = {tid: round(reverse_st[tid] - st, 2) for tid, (vm, idx), st, ft in AL}
    return deltaT, reverse_st

def total_cost(task_map, resource_pool):
    """
    Compute total cost for a task map.
    Args:
        task_map: Dictionary of (task_id, idx) to (vm_type, st, ft).
        resource_pool: List of VM types.
    Returns:
        Total cost in dollars.
    """
    vm_usage = {}
    for (tid, idx), (vm_type, st, ft) in task_map.items():
        vm_usage[(vm_type, idx)] = max(vm_usage.get((vm_type, idx), 0), ft)
    return sum(math.ceil(max_ft / 3600.0) * next(v.cost_per_interval for v in resource_pool if v.type_id == vm_type)
               for (vm_type, idx), max_ft in vm_usage.items())

# Update critical_task_optimizer to accept tasks parameter
def critical_task_optimizer(AL, task_sizes, edges, vm_speeds, slack, reverse_st, tasks):
    """
    Optimize schedule by moving critical tasks to idle windows on other VMs.
    """
    task_map = {(tid, 0): (vm, st, ft) for tid, (vm, _), st, ft in AL}
    WST = max(ft for _, _, _, ft in AL)

    all_tasks = set(task_sizes.keys()).union(set(edges.keys()).union(*[set(parents.keys()) for parents in edges.values() if parents]))
    task_preds = {tid: [] for tid in all_tasks}
    task_succs = {tid: task.successors for tid, task in tasks.items()}
    for child, parent_map in edges.items():
        for parent in parent_map:
            task_preds[child].append(parent)

    path_ranks = {}
    def compute_path_rank(tid):
        if tid in path_ranks:
            return path_ranks[tid]
        if not task_succs.get(tid, []):
            path_ranks[tid] = 1
            return 1
        total_rank = 0
        for succ in task_succs[tid]:
            total_rank += compute_path_rank(succ)
        path_ranks[tid] = total_rank
        return total_rank

    for tid in all_tasks:
        if tid not in ['t_en', 't_ex']:
            compute_path_rank(tid)

    task_info = []
    for tid in task_sizes:
        if tid not in task_map:
            continue
        delta_t = slack.get(tid, float('inf'))
        path_rank = path_ranks.get(tid, 0)
        task_info.append({
            'id': tid,
            'delta_t': delta_t,
            'path_rank': path_rank,
            'size': task_sizes[tid],
            'indicator': (abs(delta_t), -path_rank, -task_sizes[tid])
        })
    task_info.sort(key=lambda x: x['indicator'])

    critical_tasks = [t['id'] for t in task_info if abs(t['delta_t']) < 1e-3]  # Relaxed to 1e-3
    task_order = [t['id'] for t in task_info]
    best_task_map = task_map.copy()
    best_makespan = WST
    flag = 0

    for i in range(len(task_order)):
        tc = task_order[i]
        if tc not in critical_tasks:
            continue
        orig_vm_tc, orig_idx_tc, orig_st_tc, orig_ft_tc = task_map[(tc, 0)]
        et_tc = task_sizes[tc] / vm_speeds[orig_vm_tc]

        for j in range(len(task_order)):
            tu = task_order[j]
            if tu == tc or tu in set().union(*[get_transitive_nodes(tc, 'successors', task_succs, task_preds), get_transitive_nodes(tc, 'predecessors', task_succs, task_preds)]):
                continue
            vm_tu, idx_tu, st_tu, ft_tu = task_map[(tu, 0)]
            et_tu = task_sizes[tu] / vm_speeds[vm_tu]
            slack_tu = slack.get(tu, 0)

            if slack_tu <= 0:
                continue

            vm_tasks = sorted([(t, st, ft) for (t, idx), (vm, st, ft) in task_map.items() if vm == vm_tu and idx == idx_tu], key=lambda x: x[1])
            idle_windows = []
            prev_ft = 0
            for t, st, ft in vm_tasks:
                if st > prev_ft + 1e-3:  # Relaxed to 1e-3
                    idle_windows.append((prev_ft, st))
                prev_ft = ft
            if prev_ft < WST + 1e-3:
                idle_windows.append((prev_ft, WST))

            for window_st, window_ft in idle_windows:
                IT_tc = max(window_st, max([task_map[(pred, 0)][2] + (edges.get(tc, {}).get(pred, 0) / 20 if task_map[(pred, 0)][0] != vm_tu else 0)
                                          for pred in task_preds.get(tc, []) if pred in task_map] + [0]))
                OT_tc = min(window_ft, min([reverse_st.get(succ, WST) - (edges.get(succ, {}).get(tc, 0) / 20 if task_map.get((succ, 0), (vm_tu, 0, 0))[0] != vm_tu else 0)
                                          for succ in task_succs.get(tc, []) if succ in task_map] + [float('inf')]))
                if IT_tc + et_tc <= OT_tc:
                    new_st_tc = IT_tc
                    new_ft_tc = new_st_tc + et_tc
                    delay_tu = max(0, new_ft_tc - st_tu)
                    if delay_tu <= slack_tu:
                        temp_map = task_map.copy()
                        temp_map[(tc, 0)] = (vm_tu, new_st_tc, new_ft_tc)
                        if delay_tu > 0:
                            temp_map[(tu, 0)] = (vm_tu, new_ft_tc, new_ft_tc + et_tu)

                        def propagate_changes(tid, current_vm, current_st, current_ft):
                            et = task_sizes[tid] / vm_speeds[current_vm]
                            new_st = max(current_st, max([temp_map.get((pred, 0), task_map[(pred, 0)])[2] + 
                                                        (edges.get(tid, {}).get(pred, 0) / 20 if temp_map.get((pred, 0), task_map[(pred, 0)])[0] != current_vm else 0)
                                                        for pred in task_preds.get(tid, []) if pred in task_map] + [0]))
                            new_ft = new_st + et
                            temp_map[(tid, 0)] = (current_vm, new_st, new_ft)
                            for succ in task_succs.get(tid, []):
                                if succ in task_map and succ != tc:
                                    propagate_changes(succ, temp_map[(succ, 0)][0], temp_map[(succ, 0)][1], temp_map[(succ, 0)][2])

                        propagate_changes(tc, vm_tu, new_st_tc, new_ft_tc)
                        if tu != tc and delay_tu > 0:
                            propagate_changes(tu, vm_tu, new_ft_tc, new_ft_tc + et_tu)

                        vm_tasks_new = [(t, st, ft) for (t, idx), (vm, st, ft) in temp_map.items() if vm == vm_tu and idx == idx_tu]
                        overlap = any(new_st_tc < ft and new_ft_tc > st for t, st, ft in vm_tasks_new if t not in (tc, tu)) or \
                                  any((new_ft_tc if delay_tu > 0 else st_tu) < ft and (new_ft_tc + et_tu if delay_tu > 0 else ft_tu) > st for t, st, ft in vm_tasks_new if t not in (tc, tu))
                        if not overlap:
                            makespan = max(ft for _, _, ft in temp_map.values())
                            if makespan < best_makespan or (makespan == best_makespan and total_cost(temp_map, resource_pool) < total_cost(best_task_map, resource_pool)):
                                best_makespan = makespan
                                best_task_map = temp_map
                                flag = 1

    return [(tid, vm, st, ft) for (tid, idx), (vm, st, ft) in best_task_map.items()] if flag else AL

def get_transitive_nodes(tid, direction='successors', task_succs=None, task_preds=None):
    """
    Get transitive nodes in the specified direction.
    Args:
        tid: Task ID.
        direction: 'successors' or 'predecessors'.
        task_succs: Dictionary of task_id to successor list.
        task_preds: Dictionary of task_id to predecessor list.
    Returns:
        Set of transitive node IDs.
    """
    result = set()
    stack = [tid]
    while stack:
        node = stack.pop()
        if node in result or node in ['t_en', 't_ex']:
            continue
        result.add(node)
        next_nodes = task_succs.get(node, []) if direction == 'successors' else task_preds.get(node, [])
        stack.extend(next_nodes)
    result.discard(tid)
    return result

def best(S1, S2, budget, resource_pool, weight_makespan=0.9):  # Increased to 0.9
    """
    Select the best solution based on makespan and cost.
    """
    M1, AL1 = S1
    M2, AL2 = S2
    
    def compute_cost(M, AL):
        vm_usage = {}
        for _, (vm_type, _), st, ft in AL:
            vm_usage[(vm_type, _)] = max(vm_usage.get((vm_type, _), 0), ft)
        total_cost = 0
        for (vm_type, _), max_ft in vm_usage.items():
            vm_cost = next(v.cost_per_interval for v in resource_pool if v.type_id == vm_type)
            lease_duration_hours = math.ceil(max_ft / 3600.0)
            total_cost += lease_duration_hours * vm_cost
        return total_cost
    
    def compute_makespan(AL):
        return max(ft for _, _, _, ft in AL)
    
    wsc_s1 = compute_cost(M1, AL1)
    wsc_s2 = compute_cost(M2, AL2)
    wst_s1 = compute_makespan(AL1)
    wst_s2 = compute_makespan(AL2)
    s1_feasible = wsc_s1 <= budget
    s2_feasible = wsc_s2 <= budget
    
    score_s1 = (weight_makespan * wst_s1 + (1 - weight_makespan) * wsc_s1) if s1_feasible else float('inf')
    score_s2 = (weight_makespan * wst_s2 + (1 - weight_makespan) * wsc_s2) if s2_feasible else float('inf')
    
    print(f"Score S1: {score_s1}, WST: {wst_s1}, Cost: {wsc_s1}, Feasible: {s1_feasible}")
    print(f"Score S2: {score_s2}, WST: {wst_s2}, Cost: {wsc_s2}, Feasible: {s2_feasible}")
    
    return S1 if score_s1 <= score_s2 else S2

def validate_schedule(AL, edges, budget, resource_pool):
    """
    Validate the schedule for overlaps, dependencies, and budget compliance.
    Args:
        AL: List of (task_id, (vm_type, instance_id), ST, FT) tuples.
        edges: Dictionary of edge weights.
        budget: Budget in dollars per hour.
        resource_pool: List of VM types.
    Returns:
        Tuple (is_valid, message) indicating validity and any issues.
    """
    vm_schedules = {}
    for task_id, (vm_type, instance_id), st, ft in AL:
        vm_id = (vm_type, instance_id)
        if vm_id not in vm_schedules:
            vm_schedules[vm_id] = []
        vm_schedules[vm_id].append((st, ft))

    # Check for overlaps
    for vm_id, schedule in vm_schedules.items():
        sorted_schedule = sorted(schedule)
        for i in range(1, len(sorted_schedule)):
            if sorted_schedule[i][0] < sorted_schedule[i-1][1]:
                return False, f"Overlap detected on VM {vm_id} between {sorted_schedule[i-1]} and {sorted_schedule[i]}"

    # Check dependencies
    task_map = {(tid, 0): (vm, st, ft) for tid, (vm, _), st, ft in AL}
    for child_id, parents in edges.items():
        if child_id not in task_map:
            continue
        child_st, child_ft = task_map[(child_id, 0)][1], task_map[(child_id, 0)][2]
        for parent_id in parents:
            if parent_id in task_map:
                parent_ft = task_map[(parent_id, 0)][2]
                if parent_ft > child_st:
                    return False, f"Dependency violation: {parent_id} (FT: {parent_ft}) must finish before {child_id} (ST: {child_st})"

    # Check budget
    vm_usage = {}
    for _, (vm_type, _), st, ft in AL:
        vm_usage[(vm_type, _)] = max(vm_usage.get((vm_type, _), 0), ft)
    total_cost = sum(math.ceil(max_ft / 3600) * next(v.cost_per_interval for v in resource_pool if v.type_id == vm_type)
                     for (vm_type, _), max_ft in vm_usage.items())
    if total_cost > budget:
        return False, f"Budget exceeded: Cost {total_cost} > Budget {budget}"

    return True, "Schedule is valid"

def print_schedule(AL, filename="schedule_output.txt"):
    """
    Print the schedule to a file in a formatted manner.
    Args:
        AL: List of (task_id, (vm_type, instance_id), ST, FT) tuples.
        filename: Output file name.
    """
    with open(filename, 'w') as f:
        f.write("Task ID\tVM Type\tInstance ID\tStart Time\tFinish Time\n")
        for task_id, (vm_type, instance_id), st, ft in sorted(AL, key=lambda x: x[3]):  # Sort by finish time
            f.write(f"{task_id}\t{vm_type}\t{instance_id}\t{st}\t{ft}\n")
    print(f"Schedule written to {filename}")

# Update PACP_HEFT to pass tasks to critical_task_optimizer
def PACP_HEFT(T, E, R, B):
    global tasks, P0, vm_schedules, resource_pool
    tasks = {tid: task for tid, task in T.items() if tid not in ['t_en', 't_ex']}
    RTL = rtl(R)
    S_best = None
    vm_schedules = {}
    resource_pool = R
    max_iterations = 1
    
    while RTL:
        M = LeaseVM(RTL, B)
        P0 = InitializePriority(tasks, E, R, bandwidth=20)
        S0 = BuildSolution(tasks, E, M, P0, R, bandwidth=20, T=3600)
        vm_schedules.update({(vm_type, idx): sched for vm_type, idx in M for sched in [[]]})
        if S0[1]:
            print("Initial Makespan (WST0):", max(ft for _, _, _, ft in S0[1]))
            is_valid, message = validate_schedule(S0[1], E, B, R)
            print(f"Validation: {message}")
        else:
            print("Warning: No tasks scheduled in initial solution.")

        iteration_count = 0
        while iteration_count < max_iterations:
            better = False
            for cat_phase in [0, 1, 2]:
                if cat_phase == 0:
                    P1 = P0
                elif cat_phase == 1:
                    P1 = compute_thetaD(tasks, E, S0[1], bandwidth=20)
                else:
                    P1 = compute_thetaT(S0[1])

                S1 = best(S0, BuildSolution(tasks, E, M, P1, R, bandwidth=20, T=3600), B, R)
                if S1[1]:
                    print(f"Makespan after CAT Phase {cat_phase} (WST{cat_phase+1}):", max(ft for _, _, _, ft in S1[1]))
                    is_valid, message = validate_schedule(S1[1], E, B, R)
                    print(f"Validation: {message}")
                else:
                    print(f"Warning: No tasks scheduled after CAT Phase {cat_phase}.")

                if S1 != S0:
                    S0 = S1
                    better = True
                    # Continue to next phase even if better

            if not better:
                vm_speeds = {vm.type_id: vm.processing_capacity for vm in resource_pool}
                slack, reverse_st = compute_slack_refined(S0[1], E, bandwidth=20, vm_speeds=vm_speeds)
                task_sizes = {tid: tasks[tid].size for tid in tasks if tid in slack}
                AL3 = critical_task_optimizer(S0[1], task_sizes, E, vm_speeds, slack, reverse_st, tasks)
                S3 = (S0[0], AL3)
                if S3[1]:
                    print("Makespan after Optimization (WST3):", max(ft for _, _, _, ft in S3[1]))
                    is_valid, message = validate_schedule(S3[1], E, B, R)
                    print(f"Validation: {message}")
                else:
                    print("Warning: No tasks scheduled after optimization.")

                if S3 != S0:
                    S0 = S3
                    better = True

            iteration_count += 1  # Continue even if no improvement

        S_best = best(S0, S_best, B, R) if S_best else S0
        RTL = RTL[1:]

    if S_best and S_best[1]:
        print("Final Makespan (WST):", max(ft for _, _, _, ft in S_best[1]))
        is_valid, message = validate_schedule(S_best[1], E, B, R)
        print(f"Final Validation: {message}")
        print_schedule(S_best[1])
    else:
        print("Warning: No valid solution found.")

if __name__ == "__main__":
    dag_file_path = "CYBERSHAKE.n.100.4.dag"
    budget = 9  # Budget in dollars per hour
    bandwidth = 20  # MB/s
    T = 3600  # Billing interval in seconds (1 hour)
    resource_pool = [
        VM("m1.small", processing_capacity=1, cost_per_interval=0.044),  # Cost in $/hour
        VM("m1.medium", processing_capacity=2, cost_per_interval=0.087),
        VM("m3.medium", processing_capacity=3, cost_per_interval=0.067),
        VM("m1.large", processing_capacity=4, cost_per_interval=0.175),
        VM("m3.large", processing_capacity=6.5, cost_per_interval=0.133),
        VM("m1.xlarge", processing_capacity=8, cost_per_interval=0.350),
        VM("m3.xlarge", processing_capacity=13, cost_per_interval=0.266),
        VM("m3.2xlarge", processing_capacity=26, cost_per_interval=0.532)
    ]
    
    tasks, edges = parse_dag_file(dag_file_path)
    task_successors = {tid: task.successors for tid, task in tasks.items()}
    result = PACP_HEFT(tasks, edges, resource_pool, budget)