# VM speeds and bandwidth
vm_speeds = {"VM0": 2, "VM1": 1}
bandwidth = 1  # MB/s

# Initial schedule used to compute θD
initial_schedule = [
    ['t0', 'VM0', 0, 10],
    ['t1', 'VM0', 10, 41],
    ['t2', 'VM1', 11, 41],
    ['t3', 'VM0', 70, 100],
    ['t4', 'VM0', 41, 43],
    ['t5', 'VM0', 43, 70],
    ['t6', 'VM0', 100, 105]
]

# Task sizes
task_sizes = {
    "t0": 20.0,
    "t1": 62.0,
    "t2": 30.0,
    "t3": 60.0,
    "t4": 4.0,
    "t5": 54.0,
    "t6": 10.0
}

# Edge dependencies: child -> {parent: data_size_MB}
edges = {
    "t1": {"t0": 1.0},
    "t2": {"t0": 1.0},
    "t3": {"t0": 1.0},
    "t4": {"t0": 1.0},
    "t5": {"t4": 1.0},
    "t6": {"t1": 3.0, "t2": 2.0, "t3": 2.0, "t5": 2.0}
}

# Mapping from initial schedule
task_vm = {tid: vm for tid, vm, _, _ in initial_schedule}
task_et = {tid: ft - st for tid, _, st, ft in initial_schedule}

# Successors mapping
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

# Priority1 from θD
priority1 = {tid: i for i, (tid, _) in enumerate(sorted(thetaD.items(), key=lambda x: -x[1]))}

# Build AL function
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

# Build AL1 from Priority1
AL1 = build_AL(priority1, task_sizes, edges, vm_speeds, bandwidth)
WST1 = max(ft for _, _, _, ft in AL1)

# θT from AL1
thetaT = {tid: WST1 - st for tid, _, st, _ in AL1}
priority2 = {tid: i for i, (tid, _) in enumerate(sorted(thetaT.items(), key=lambda x: -x[1]))}

# Build AL2 from Priority2
AL2 = build_AL(priority2, task_sizes, edges, vm_speeds, bandwidth)
WST2 = max(ft for _, _, _, ft in AL2)

# Results
print("θD values:")
for tid, val in thetaD.items():
    print(f"{tid}: {val:.2f}")

print("\nPriority1 (θD):")
for tid, rank in priority1.items():
    print(f"{tid}: {rank}")

print("\nAL1 (based on θD):")
print("Task\tVM\tST\tFT\tET")
for tid, vm, st, ft in AL1:
    et = round(ft - st, 2)
    print(f"{tid}\t{vm}\t{st}\t{ft}\t{et}")
print(f"→ Makespan (AL1): {WST1} seconds")

print("\nθT values:")
for tid, val in thetaT.items():
    print(f"{tid}: {val:.2f}")

print("\nPriority2 (θT):")
for tid, rank in priority2.items():
    print(f"{tid}: {rank}")

print("\nAL2 (based on θT):")
print("Task\tVM\tST\tFT\tET")
for tid, vm, st, ft in AL2:
    et = round(ft - st, 2)
    print(f"{tid}\t{vm}\t{st}\t{ft}\t{et}")
print(f"→ Makespan (AL2): {WST2} seconds")
