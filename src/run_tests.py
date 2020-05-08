from multiprocessing import Pool
import subprocess

def accelerator_sizes():
    # Tuples contain: (width, height)
    options = [(1,1), (2,2), (4,4), (8,8), (2,4), (4,2)]

    for (w,h) in options:
        yield [f"--accelerator-pe-columns={w}", f"--accelerator-pe-rows={h}"]

def connectivity(dimension):
    # Not all connectivity options make sense. e.g. It doesn't make sense to
    # broadcast inputs along one dimension and accumulate outputs along the same
    # dimension.
    # Tuples contain: (broadcast?, accumulate?)
    options = [(0,0), (1,0), (0,1)]

    for (bcast, acc) in options:
        yield [f"--accelerator-broadcast-{dimension}={bcast}", f"--accelerator-accumulate-{dimension}={acc}"]

def timings():
    # NOTE: Not yet implemented in the simulator.
    # Tuples contain: (latency, initiation interval) of accelerator PEs
    options = [(1,1)]

    for (latency, ii) in options:
        yield [f"--accelerator-pe-latency={latency}", f"--accelerator-pe-initiation-interval={ii}"]

def memory_config():
    # TODO: add more options for numbers of banks, bank sizes, etc.
    # Currently the simulator can't handle those options well.
    # Options are currently just magic memory flags.
    options = [0, 1]

    for magic in options:
        yield [f"--magic-memory={magic}"]

def all_options():
    for size in accelerator_sizes():
        for row in connectivity("rows"):
            for col in connectivity("columns"):
                for timing in timings():
                    for memory in memory_config():
                        yield size + row + col + timing + memory

def run_experiment(params):
    cmd = ["lokisim"] + params + ["build/lat-test"]
    test = subprocess.run(cmd, capture_output=True, timeout=10)
    return test


if __name__ == "__main__":
    with Pool() as p:
        results = p.map(run_experiment, all_options())

    passed = []
    failed = []

    for test in results:
        if test.returncode == 0:
            passed.append(" ".join(test.args))
        else:
            failed.append(" ".join(test.args))

    # TODO: add a "Jenkins mode" which outputs an appropriately-formatted XML
    # file. 
    print(f"Passed {len(passed)}, failed {len(failed)}")

    with open("passed.txt", mode="w") as f:
        for test in passed:
            f.write(test + "\n")

    with open("failed.txt", mode="w") as f:
        for test in failed:
            f.write(test + "\n")
