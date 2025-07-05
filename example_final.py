import math
import collections

class Task:
    def __init__(self, id, size):
        self.id = id
        self.size = size
        self.predecessors = []
        self.successors = []

# VM speeds and bandwidth
vm_speeds = {"VM0": 2, "VM1": 1}
bandwidth = 1  # MB/s

# Task sizes
task_sizes = {
    "t0": 20.0, "t1": 62.0, "t2": 30.0, "t3": 60.0,
    "t4": 4.0, "t5": 54.0, "t6": 10.0
}

# Edge dependencies: child -> {parent: data_size_MB}
edges = {
    "t1": {"t0": 1.0}, "t2": {"t0": 1.0}, "t3": {"t0": 1.0},
    "t4": {"t0": 1.0}, "t5": {"t4": 1.0},
    "t6": {"t1": 3.0, "t2": 2.0, "t3": 2.0, "t5": 2.0}
}

# Build task relationships
tasks = {tid: Task(tid, size) for tid, size in task_sizes.items()}
for child, parent_dict in edges.items():
    for parent in parent_dict:
        tasks[child].predecessors.append(parent)
        tasks[parent].successors.append(child)

# Initial schedule used to compute θD
initial_schedule = [
    ['t0', 'VM0', 0, 10], ['t1', 'VM0', 10, 41],
    ['t2', 'VM1', 11, 41], ['t3', 'VM0', 70, 100],
    ['t4', 'VM0', 41, 43], ['t5', 'VM0', 43, 70],
    ['t6', 'VM0', 100, 105]
]

# Mapping from initial schedule
task_vm = {tid: vm for tid, vm, _, _ in initial_schedule}
task_et = {tid: ft - st for tid, _, st, ft in initial_schedule}

task_successors = {tid: [] for tid in task_sizes}
for child, parent_dict in edges.items():
    for parent in parent_dict:
        task_successors[parent].append(child)

# θD calculation
thetaD = {}
def compute_thetaD(tid):
    if tid in thetaD:
        return thetaD[tid]
    if not task_successors[tid]:
        thetaD[tid] = task_et[tid]
        return thetaD[tid]
    max_val = 0
    for succ in task_successors[tid]:
        et = task_et[tid]
        tt = edges.get(succ, {}).get(tid, 0)
        if task_vm[tid] != task_vm[succ]:
            tt = tt / bandwidth
        else:
            tt = 0
        val = compute_thetaD(succ) + tt + et
        max_val = max(max_val, val)
    thetaD[tid] = max_val
    return max_val

for tid in task_sizes:
    compute_thetaD(tid)

priority1 = {tid: i for i, (tid, _) in enumerate(sorted(thetaD.items(), key=lambda x: -x[1]))}

def build_AL(priority, task_sizes, edges, vm_speeds, bandwidth):
    vm_ready = {vm: 0 for vm in vm_speeds}
    task_ft = {}
    task_vm_alloc = {}
    AL = []

    task_preds = {tid: [] for tid in task_sizes}
    for child, parent_dict in edges.items():
        for parent in parent_dict:
            task_preds[child].append(parent)

    task_order = [tid for tid, _ in sorted(priority.items(), key=lambda x: x[1])]

    for tid in task_order:
        best_vm = None
        best_est = 0
        best_ft = float("inf")
        size = task_sizes[tid]

        for vm, speed in vm_speeds.items():
            et = size / speed
            est = 0
            for pred in task_preds.get(tid, []):
                pred_ft = task_ft[pred]
                pred_vm = task_vm_alloc[pred]
                tt = edges.get(tid, {}).get(pred, 0) / bandwidth if pred_vm != vm else 0
                est = max(est, pred_ft + tt)
            est = max(est, vm_ready[vm])
            ft = est + et
            if ft < best_ft:
                best_vm, best_est, best_ft = vm, est, ft

        AL.append([tid, best_vm, round(best_est, 2), round(best_ft, 2)])
        task_ft[tid] = best_ft
        task_vm_alloc[tid] = best_vm
        vm_ready[best_vm] = best_ft

    return AL

AL1 = build_AL(priority1, task_sizes, edges, vm_speeds, bandwidth)
WST1 = max(ft for _, _, _, ft in AL1)

thetaT = {tid: WST1 - st for tid, _, st, _ in AL1}
priority2 = {tid: i for i, (tid, _) in enumerate(sorted(thetaT.items(), key=lambda x: -x[1]))}

AL2 = build_AL(priority2, task_sizes, edges, vm_speeds, bandwidth)
WST2 = max(ft for _, _, _, ft in AL2)

def compute_slack_refined(AL, edges, bandwidth):
    task_map = {tid: (vm, st, ft) for tid, vm, st, ft in AL}
    WST = max(ft for _, _, _, ft in AL)
    reverse_st = {}
    reverse_ft = {tid: WST for tid in task_map}

    tasks_sorted = sorted(AL, key=lambda x: -x[3])
    fpt = {vm: WST for vm in vm_speeds}
    succ_map = {tid: [] for tid in task_map}
    for child, parent_map in edges.items():
        for parent in parent_map:
            if parent in succ_map:
                succ_map[parent].append(child)

    for tid, vm, _, _ in tasks_sorted:
        succs = succ_map.get(tid, [])
        min_succ_start = float('inf')
        for succ in succs:
            if succ not in task_map:
                continue
            succ_vm, succ_st, _ = task_map[succ]
            tt = edges.get(succ, {}).get(tid, 0) / bandwidth if succ_vm != vm else 0
            min_succ_start = min(min_succ_start, reverse_st.get(succ, WST) - tt)
        min_succ_start = min(min_succ_start, fpt[vm])
        et = task_map[tid][2] - task_map[tid][1]
        reverse_ft[tid] = min_succ_start
        reverse_st[tid] = min_succ_start - et
        fpt[vm] = reverse_st[tid]
    deltaT = {tid: round(reverse_st[tid] - task_map[tid][1], 2) for tid in task_map}
    return deltaT,reverse_st

slack,reverse_st = compute_slack_refined(AL2, edges, bandwidth)

def critical_task_optimizer(AL, task_sizes, edges, vm_speeds, slack, reverse_st):
    print("Step 1: Computing slack times (ΔT) using reverse scheduling...")
    task_map = {tid: (vm, st, ft) for tid, vm, st, ft in AL}
    WST = max(ft for _, _, _, ft in AL)

    # Sort tasks by deltaT ascending (lower slack = higher criticality) and assign RT
    print("Step 2: Ranking tasks by ΔT (ascending) for criticality...")
    tasks_sorted_by_delta = sorted(task_sizes.keys(), key=lambda t: -slack.get(t, float('inf')))
    path_id = {}
    current_rt = 1
    last_delta = None
    for tid in tasks_sorted_by_delta:
        current_slack = slack.get(tid, float('inf'))
        if last_delta is None or abs(current_slack - last_delta) > 1e-10:
            current_rt += 0 if last_delta is None else 1
        path_id[tid] = current_rt
        last_delta = current_slack
    # Adjust ranks to start from 1 and handle ties
    min_rank = min(path_id.values())
    path_id = {tid: rank - min_rank + 1 for tid, rank in path_id.items()}
    # Group tasks with same slack to same rank
    rank_groups = {}
    for tid, rank in path_id.items():
        rank_groups.setdefault(slack.get(tid, float('inf')), []).append(tid)
    for slack_val, tids in rank_groups.items():
        if len(tids) > 1:
            new_rank = path_id[tids[0]]
            for tid in tids:
                path_id[tid] = new_rank

    # Compute transitive predecessors and successors
    def get_transitive_nodes(tid, direction='successors'):
        result = set()
        stack = [tid]
        while stack:
            node = stack.pop()
            if node in result or node in ['t_en', 't_ex']:
                continue
            result.add(node)
            next_nodes = task_successors.get(node, []) if direction == 'successors' else task_preds.get(node, [])
            stack.extend(next_nodes)
        result.discard(tid)
        return result
    
    task_preds = {tid: [] for tid in task_sizes}
    for child, parent_dict in edges.items():    
        for parent in parent_dict:
            task_preds[child].append(parent)

    # Compute phi (omega): sum of ranks of all transitive nodes
    print("Step 3: Computing Φ (omega) for tasks as sum of transitive nodes' ranks...")
    phi = {}
    print(path_id)
    for tid in task_sizes:
        transitive_preds = get_transitive_nodes(tid, 'predecessors')
        transitive_succs = get_transitive_nodes(tid, 'successors')
        transitive_nodes = transitive_preds.union(transitive_succs)
        print(tid , transitive_nodes)
        phi[tid] = sum(path_id.get(node, 0) for node in transitive_nodes if node in path_id)    
        #phi = {'t0': 11.0, 't1': 8.0, 't2': 4.0, 't3': 4.0, 't4': 6.0, 't5': 6.0, 't6': 11.0}

    # Criticality sort based on slack, phi, and -size
    print("Step 3.5: Performing criticality sort (ascending slack, descending phi, ascending -size)...")
    task_info = []
    for tid in task_sizes:
        if tid not in slack:
            continue
        indicator = slack[tid]  # Ascending slack (lower is better)
        task_info.append({
            'id': tid,
            'delta_t': slack[tid],
            'phi': phi.get(tid, 0.0),  # Higher phi is better
            'size': task_sizes[tid],   # Smaller -size is better (negative size)
            'indicator': (slack[tid], -phi.get(tid, 0.0), -task_sizes[tid])  # Tuple for sorting
        })
        
    # Sort ascending by slack, then descending by phi, then ascending by size
    task_info.sort(key=lambda x: x['indicator'])
    # print("task_info",task_info)

    print("\nTask Rankings after Criticality Sort:")
    print("Task\tdeltaT\tΦ\tSize\tRank")
    for rank, t in enumerate(task_info):
        print(f"{t['id']}\t{t['delta_t']}\t{t['phi']:.2f}\t{t['size']}\t{rank}")

def critical_task_optimizer(AL, task_sizes, edges, vm_speeds, slack, reverse_st):
    print("Step 1: Computing slack times (ΔT) using reverse scheduling...")
    task_map = {tid: (vm, st, ft) for tid, vm, st, ft in AL}
    WST = max(ft for _, _, _, ft in AL)

    # Compute transitive predecessors and successors
    def get_transitive_nodes(tid, direction='successors'):
        result = set()
        stack = [tid]
        while stack:
            node = stack.pop()
            if node in result or node in ['t_en', 't_ex']:
                continue
            result.add(node)
            next_nodes = task_successors.get(node, []) if direction == 'successors' else task_preds.get(node, [])
            stack.extend(next_nodes)
        result.discard(tid)
        return result
    
    task_preds = {tid: [] for tid in task_sizes}
    for child, parent_dict in edges.items():    
        for parent in parent_dict:
            task_preds[child].append(parent)

    # Rank tasks by slack, then sum of successor ranks, then -size
    print("Step 2: Ranking tasks by ΔT (ascending), ΣRᵀ (descending), and -size (ascending)...")
    task_info = []
    for tid in task_sizes:
        if tid not in task_map:
            continue
        successors = task_successors.get(tid, [])
        successor_ranks = sum(1 for succ in successors if succ in task_map)  # Simplified ΣRᵀ
        task_info.append({
            'id': tid,
            'delta_t': slack.get(tid, float('inf')),
            'successor_ranks': successor_ranks,
            'size': task_sizes[tid],
            'indicator': (slack.get(tid, float('inf')), -successor_ranks, -task_sizes[tid])
        })
    task_info.sort(key=lambda x: x['indicator'])

    print("\nTask Rankings after Criticality Sort:")
    print("Task\tdeltaT\tΣRᵀ\tSize\tRank")
    for rank, t in enumerate(task_info):
        print(f"{t['id']}\t{t['delta_t']}\t{t['successor_ranks']}\t{t['size']}\t{rank}")

    # Critical task optimization
    print("Step 4: Searching for an optimization...")
    critical_tasks = [t for t in slack if abs(slack[t]) < 1e-10]
    print(f"Critical tasks: {critical_tasks}")
    task_order = [t['id'] for t in task_info]
    best_task_map = task_map.copy()
    best_makespan = WST
    print("Trying task-VM assignments:")
    flag = 0

    for i in range(len(task_order)):
        tc = task_order[i]  # Critical task
        if tc not in critical_tasks:
            continue
        for j in range(len(task_order)):
            tu = task_order[j]  # Non-critical task
            if tu == tc or tu in get_transitive_nodes(tc, 'successors') or tu in get_transitive_nodes(tc, 'predecessors'):
                continue
            vm_tu = task_map[tu][0]
            current_st_tu = task_map[tu][1]
            current_ft_tu = task_map[tu][2]
            et_tu = current_ft_tu - current_st_tu
            slack_tu = slack.get(tu, 0)
            if slack_tu <= 0:
                continue

            # Compute idle window [x_tu, y_tu] with reverse schedule context
            vm_tasks = sorted([(t, st, ft) for t, (v, st, ft) in task_map.items() if v == vm_tu], key=lambda x: x[2])
            x_tu = max([ft for t, _, ft in vm_tasks if ft <= reverse_st[tu]] + [0])
            y_tu = reverse_st[tu]
            print(f"Debug: Task '{tu}' on VM '{vm_tu}', x_tu={x_tu:.2f}, y_tu={y_tu:.2f}, slack={slack_tu:.2f}, ET_tu={et_tu:.2f}")

            # IT and OT for tc on vm_tu
            IT_tc = max([task_map[pred][2] + (edges.get(tc, {}).get(pred, 0) / bandwidth if task_map[pred][0] != vm_tu else 0)
                        for pred in task_preds.get(tc, []) if pred in task_map] + [0])
            OT_tc = min([reverse_st[succ] - (edges.get(succ, {}).get(tc, 0) / bandwidth if task_map[succ][0] != vm_tu else 0)
                        for succ in task_successors.get(tc, []) if succ in task_map] + [float('inf')])
            print(f"Debug: Task '{tc}', IT_tc={IT_tc:.2f}, OT_tc={OT_tc:.2f}")

            # Earliest start and latest finish for tc
            EST_tc = max(x_tu, IT_tc)
            LFT_tc = min(y_tu, OT_tc)
            et_tc = task_sizes[tc] / vm_speeds[vm_tu]
            print(f"Debug: EST_tc={EST_tc:.2f}, LFT_tc={LFT_tc:.2f}, et_tc={et_tc:.2f}")

            if EST_tc + et_tc <= LFT_tc:
                # Delay tu to start after tc's new finish time
                new_st_tu = EST_tc + et_tc
                new_ft_tu = new_st_tu + et_tu
                delay = new_ft_tu - current_ft_tu
                if delay <= slack_tu:
                    temp_map = task_map.copy()
                    temp_map[tc] = (vm_tu, EST_tc, EST_tc + et_tc)
                    temp_map[tu] = (vm_tu, new_st_tu, new_ft_tu)
                    print(f"Debug: Proposed - tc='{tc}' to [{EST_tc:.2f}-{EST_tc + et_tc:.2f}], tu='{tu}' to [{new_st_tu:.2f}-{new_ft_tu:.2f}]")

                    # Propagate on original VM of tc
                    orig_vm_tc = [vm for t, (vm, _, _) in task_map.items() if t == tc][0]
                    if orig_vm_tc != vm_tu:
                        vm_tasks_orig = sorted([(t, st, ft) for t, (v, st, ft) in task_map.items() if v == orig_vm_tc], key=lambda x: x[1])
                        last_finish = max([ft for t, _, ft in vm_tasks_orig if ft <= task_map[tc][1]] + [0])
                        current_time = last_finish
                        for t, st, ft in vm_tasks_orig:
                            if t == tc or st < task_map[tc][1]:
                                continue
                            et_task = task_sizes[t] / vm_speeds[orig_vm_tc]
                            earliest_start = max(current_time, max([temp_map.get(pred, task_map[pred])[2] + (edges.get(t, {}).get(pred, 0) / bandwidth if temp_map.get(pred, task_map[pred])[0] != orig_vm_tc else 0)
                                                                  for pred in task_preds.get(t, []) if pred in task_map] + [0]))
                            temp_map[t] = (orig_vm_tc, earliest_start, earliest_start + et_task)
                            current_time = earliest_start + et_task

                    # Verify no overlap on vm_tu
                    vm_tasks = [(t, st, ft) for t, (v, st, ft) in task_map.items() if v == vm_tu and t not in (tc, tu)]
                    overlap = any(EST_tc < ft and EST_tc + et_tc > st for t, st, ft in vm_tasks) or \
                              any(new_st_tu < ft and new_ft_tu > st for t, st, ft in vm_tasks)
                    if not overlap:
                        makespan = max(ft for _, _, ft in temp_map.values())
                        print(f"Debug: temp_map = {dict(temp_map)}")
                        print(f"  > Tried moving Task '{tc}' to [{EST_tc:.2f}-{EST_tc + et_tc:.2f}] and delaying '{tu}' to [{new_st_tu:.2f}-{new_ft_tu:.2f}] with makespan {makespan:.2f}")
                        if makespan < best_makespan or not flag:
                            best_makespan = makespan
                            best_task_map = temp_map.copy()
                            print(f"  > Found better move: Moving Task '{tc}' to [{EST_tc:.2f}-{EST_tc + et_tc:.2f}] and delaying '{tu}' to [{new_st_tu:.2f}-{new_ft_tu:.2f}] with makespan {makespan:.2f}")
                            flag = 1
                    else:
                        print(f"Debug: Overlap detected for tc='{tc}' or tu='{tu}'")
                else:
                    print(f"Debug: Delay {delay:.2f} exceeds slack {slack_tu:.2f} for tu='{tu}'")
            else:
                print(f"Debug: No fit - EST_tc + et_tc ({EST_tc + et_tc:.2f}) > LFT_tc ({LFT_tc:.2f}) for tc='{tc}'")

    if flag == 0:
        print("Step 5: No optimization found. Failed to optimize critical tasks.")
    else:
        print("Step 5: Optimization found. Building new solution AL3...")
    print(sorted([(tid, vm, st, ft) for tid, (vm, st, ft) in best_task_map.items()], key=lambda x: reverse_st[x[0]]))
    return sorted([(tid, vm, st, ft) for tid, (vm, st, ft) in best_task_map.items()], key=lambda x: reverse_st[x[0]])
    

AL3 = critical_task_optimizer(AL2, task_sizes, edges, vm_speeds, slack,reverse_st)
WST3 = max(ft for _, _, _, ft in AL3)


print("\nθD values:")
for tid, val in sorted(thetaD.items()):
    print(f"{tid}: {val:.2f}")

print("\nPriority1 (θD):")
for tid, rank in sorted(priority1.items()):
    print(f"{tid}: {rank}")

print("\nAL1 (based on θD):")
print("Task\tVM\tST\tFT\tET")
for tid, vm, st, ft in AL1:
    et = round(ft - st, 2)
    print(f"{tid}\t{vm}\t{st}\t{ft}\t{et}")
print(f"→ Makespan (AL1): {WST1} seconds")

print("\nθT values:")
for tid, val in sorted(thetaT.items()):
    print(f"{tid}: {val:.2f}")

print("\nPriority2 (θT):")
for tid, rank in sorted(priority2.items()):
    print(f"{tid}: {rank}")

print("\nAL2 (based on θT):")
print("Task\tVM\tST\tFT\tET")
for tid, vm, st, ft in AL2:
    et = round(ft - st, 2)
    print(f"{tid}\t{vm}\t{st}\t{ft}\t{et}")
print(f"→ Makespan (AL2): {WST2} seconds")

print("\nSlack Time ΔT (Delta T) for Each Task:")
for tid in sorted(slack):
    print(f"{tid}: {slack[tid]}")

print("\nAL3 (after Critical Task Optimization):")
print("Task\tVM\tST\tFT\tET")
for tid, vm, st, ft in AL3:
    et = round(ft - st, 2)
    print(f"{tid}\t{vm}\t{st}\t{ft}\t{et}")
print(f"→ Makespan (AL3): {WST3} seconds")