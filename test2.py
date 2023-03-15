import time
from datetime import datetime
from prefect import flow, task
from prefect_ray import RayTaskRunner
from prefect_shell import ShellOperation
from prefect_ray.context import remote_options
import ray
import time
@task
def test1():
    print("hh")
    while True:
        print("xxx")
        time.sleep(30)

@task
def test2():
    print("hhx")
    while True:
        print("xxx1")
        time.sleep(30)

@flow(task_runner=RayTaskRunner(address='ray://10.105.48.154:10001'))
def count_to():
    with remote_options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id = '693c57a1facb7d49bc1851c1fda9095f2408539f1f7945c8ea864585',
        soft=False)
    ):
        test1.submit()
    with remote_options(
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id = 'b79c961e8ba7526242ead81a7bce3cec95fc112bb411725d4dc90241',
        soft=False)
    ):
        test2.submit()
    

if __name__ == "__main__":
    count_to()

# outputs
#3
#7
#2
#6
#4
#0
#1
#5
#8
#9

