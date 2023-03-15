import ray
import ray.experimental.state.api as api

ray.init(address='ray://10.105.48.153:10001')
# Define the square task.
@ray.remote
def node_affinity_func():
    print(type(ray.get_runtime_context().node_id))
    print(ray.get_runtime_context().node_id)
    f = open('/root/log.txt','w')
    for i in range(100):
        print('xxxx',file=f)    #print函数加了\n,不需要再加了
    f.close()
    return ray.get_runtime_context().node_id




# print(type(bytes.fromhex('bbfeba06018ca5e3a68d8a9d573488a282bc4eef69b663b920da59b0')))
# print(type(ray.get_runtime_context().node_id))
# print(ray.get_runtime_context().node_id)
# Only run the task on the local node.
# nodes = api.list_nodes()
# print(nodes)

ray.get(node_affinity_func.options(
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        #node_id='18b7c039006827ac2b461915c9005e39ab01b5910482c3bbf886b5a1',
        node_id = '3c5808c7d99526b891e2b9c8d08ab6cd78a431c9af36ac8ea6121f3c',
        soft=False,
    )
).remote())

