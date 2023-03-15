from typing import List
import ray
# <follow the previous code>
from ray import workflow
# Define Ray remote functions.
# ray.init(address='ray://10.105.48.154:10001',storage="file:///tmp/ray/workflow_data")

ray.init(address='ray://10.105.48.154:10001')
@ray.remote
def read_data(num: int):
    return [i for i in range(num)]

@ray.remote
def preprocessing(data: List[float]) -> List[float]:
    return [d**2 for d in data]

@ray.remote
def aggregate(data: List[float]) -> float:
    return sum(data)

# Build the DAG:
# data -> preprocessed_data -> aggregate
data = read_data.bind(10)
preprocessed_data = preprocessing.bind(data)
output = aggregate.bind(preprocessed_data)


# Execute the workflow and print the result.
print(workflow.run(output))

# You can also run the workflow asynchronously and fetch the output via
# 'ray.get'
output_ref = workflow.run_async(output)
print(ray.get(output_ref))