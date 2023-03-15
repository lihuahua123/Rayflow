from prefect import flow
import logging
import os
import shutil
import time
import numpy as np
#from kafka import KafkaConsumer
import json

from typing import List
from joblib import dump, load

from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import queue
import time
from datetime import datetime
from prefect import flow, task
from prefect_ray import RayTaskRunner
from prefect_shell import ShellOperation
from prefect_ray.context import remote_options
import ray
current_dir = '/root/prefect'#os.path.dirname(__file__)
dataset_path = os.path.join(current_dir, 'dataset', 'mnist_{}.npz')
working_dir = os.path.join(current_dir, 'tmp')

trained_model_dir = os.path.join(working_dir, 'trained_models')
validated_model_dir = os.path.join(working_dir, 'validated_models')
deployed_model_dir = os.path.join(working_dir, 'deployed_models')

#model_deployed_event = queue.Queue()
other_event = queue.Queue()
def _prepare_working_dir():
    for path in [trained_model_dir, validated_model_dir, deployed_model_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)


def _get_latest_model(model_dir) -> str:
    file_list = os.listdir(model_dir)
    if file_list is None or len(file_list) == 0:
        return None
    else:
        file_list.sort(reverse=True)
        return os.path.join(model_dir, file_list[0])


def _preprocess_data_p(x_data, y_data):
    random_state = check_random_state(0)
    permutation = random_state.permutation(x_data.shape[0])
    x_train = x_data[permutation]
    y_train = y_data[permutation]

    reshaped_x_train = x_train.reshape((x_train.shape[0], -1))
    scaler_x_train = StandardScaler().fit_transform(reshaped_x_train)
    return scaler_x_train, y_train
def _preprocess_data(dataset_uri):
    with np.load(dataset_uri) as f:
        x_data, y_data = f['x_train'], f['y_train']
    return  _preprocess_data_p(x_data, y_data)

def preprocess1():
    _prepare_working_dir()
    train_dataset = dataset_path.format('train')
    try:
        event_sender = AIFlowNotificationClient(NOTIFICATION_SERVER_URI)
        my_topic = "test2"
        consumer = KafkaConsumer(my_topic, bootstrap_servers='10.105.48.156:9092', auto_offset_reset='smallest', group_id='test')
        count = 0
        x_train, y_train = [], []
        for msg in consumer: # 这里会监听，无限循环
            count += 1
            jsont = json.loads(str(msg.value,'utf-8'))
            x_train.append(jsont["xdata"] )
            y_train.append(jsont["ydata"] )
            if count % 1000 == 0:
                x_train1, y_train1 = _preprocess_data_p(np.array(x_train),np.array(y_train))
                np.save(os.path.join(working_dir, f'x_train'), x_train1) # 覆盖
                np.save(os.path.join(working_dir, f'y_train'), y_train1)
                event_sender.send_event(key="data_prepared", value=None)
                x_train, y_train = [], []
                #time.sleep(30)
    finally:
        event_sender.close()

@task(retries=3)
def preprocess(other_event):
    _prepare_working_dir()
    train_dataset = dataset_path.format('train')
    # try:
    #     event_sender = AIFlowNotificationClient(NOTIFICATION_SERVER_URI)
    #while True:
    x_train, y_train = _preprocess_data(train_dataset)
    np.save(os.path.join(working_dir, f'x_train'), x_train)
    np.save(os.path.join(working_dir, f'y_train'), y_train)
    # #event_sender.send_event(key="data_prepared", value=None)
    other_event.put("data_prepared")
    print("data prepared!!")
    time.sleep(30)
    #finally:
        #event_sender.close()
@task(retries=3)
def train():
    """
    See also:
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
    """
    _prepare_working_dir()
    clf = LogisticRegression(C=50. / 5000, penalty='l1', solver='saga', tol=0.1)
    x_train = np.load(os.path.join(working_dir, f'x_train.npy'))
    y_train = np.load(os.path.join(working_dir, f'y_train.npy'))
    clf.fit(x_train, y_train)
    model_path = os.path.join(trained_model_dir, time.strftime("%Y%m%d%H%M%S", time.localtime()))
    dump(clf, model_path)

@task(retries=3)
def validate():
    _prepare_working_dir()

    validate_dataset = dataset_path.format('evaluate')
    x_validate, y_validate = _preprocess_data(validate_dataset)

    to_be_validated = _get_latest_model(trained_model_dir)
    clf = load(to_be_validated)
    scores = cross_val_score(clf, x_validate, y_validate, scoring='precision_macro')
    try:
        #event_sender = AIFlowNotificationClient(NOTIFICATION_SERVER_URI)
        deployed_model = _get_latest_model(deployed_model_dir)
        if deployed_model is None:
            logging.info(f"Generate the 1st model with score: {scores}")
            shutil.copy(to_be_validated, validated_model_dir)
            #event_sender.send_event(key="model_validated", value=None)
            #other_event.put("model_validated")
        else:
            deployed_clf = load(deployed_model)
            old_scores = cross_val_score(deployed_clf, x_validate, y_validate, scoring='precision_macro')
            if np.mean(scores) > np.mean(old_scores):
                logging.info(f"A new model with score: {scores} passes validation")
                shutil.copy(to_be_validated, validated_model_dir)
                #event_sender.send_event(key="model_validated", value=None)
                #other_event.put("model_validated")
            else:
                logging.info(f"New generated model with score: {scores} is worse "
                             f"than the previous: {old_scores}, ignored.")
    finally:
        pass
        #event_sender.close()

@task(retries=3)
def deploy():
    _prepare_working_dir()
    to_be_deployed = _get_latest_model(validated_model_dir)
    return to_be_deployed
    #deploy_model_path = shutil.copy(to_be_deployed, deployed_model_dir)
    #model_deployed_event.put(to_be_deployed)
    # try:
    #     event_sender = AIFlowNotificationClient(NOTIFICATION_SERVER_URI)
    #     event_sender.send_event(key="model_deployed", value=deploy_model_path)
    # finally:
    #     event_sender.close()


# class ModelLoader(ListenerProcessor):
#     def __init__(self):
#         self.current_model = None
#         logging.info("Waiting for the first model deployed...")

#     def process(self, events: List[Event]):
#         for e in events:
#             self.current_model = e.value

@task(retries=3)
def predict(model):
    _prepare_working_dir()
    predict_dataset = dataset_path.format('predict')
    result_path = os.path.join(working_dir, 'predict_result')
    x_predict, _ = _preprocess_data(predict_dataset)
    

    #model_loader = ModelLoader()
    #while True:
    #model = model_deployed_event.get()
    current_model = model#model_loader.current_model
    clf = load(current_model)
    result = clf.predict(x_predict)
    with open(result_path, 'a') as f:
        f.write(f'model [{current_model}] predict result: {result}\n')
    #model_deployed_event.task_done()
    # try:
    #     event_listener = AIFlowNotificationClient(NOTIFICATION_SERVER_URI)
    #     event_listener.register_listener(listener_processor=model_loader,
    #                                      event_keys=["model_deployed", ])
    #     while True:
    #         if current_model != model_loader.current_model:
    #             current_model = model_loader.current_model
    #             logging.info(f"Predicting with new model: {current_model}")
    #             clf = load(current_model)
    #             result = clf.predict(x_predict)
    #             with open(result_path, 'a') as f:
    #                 f.write(f'model [{current_model}] predict result: {result}\n')
    #         time.sleep(5)
    # finally:
    #     event_listener.close()

@flow(name="online_machine_learning", task_runner=RayTaskRunner(address='ray://10.105.48.154:10001'))
def online_machine_learning():
    # with remote_options(
    #     scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
    #     node_id = '693c57a1facb7d49bc1851c1fda9095f2408539f1f7945c8ea864585',
    #     soft=False)
    # ):
    yid = preprocess.remote()
    while True:
        _ = other_event.get("data_prepared")
        # with remote_options(
        #     scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        #     node_id = 'b79c961e8ba7526242ead81a7bce3cec95fc112bb411725d4dc90241',
        #     soft=False)
        # ):
        train().submit()
        validate().submit()
        d = deploy().submit()
        predict(d).submit()


online_machine_learning()