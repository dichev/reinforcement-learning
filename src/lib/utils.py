from datetime import datetime
import timeit, statistics


def now():
    return datetime.now().strftime('%b%d %H-%M-%S')


def measure(name, fn, repeat=5, number=20):
    times = timeit.repeat(fn, setup='pass', repeat=repeat, number=number, globals=globals())
    mu = statistics.mean(times)
    std = statistics.stdev(times) if repeat > 1 else 0
    print(f'{name} : {mu:.8f}s ±{std:.8f} ({repeat=}, {number=})')