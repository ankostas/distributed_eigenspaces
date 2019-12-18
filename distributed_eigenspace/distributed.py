#!/usr/bin/env python

import pika
import json
import argparse
import numpy as np


class Node:
    def __init__(self, broker_host):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=broker_host))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='master')
        self.channel.queue_declare(queue='slaves')


class SlaveNode(Node):
    def __init__(self, broker_host):
        super().__init__(broker_host)
        self.channel.basic_consume(queue='slaves', on_message_callback=self.callback_)

    def start(self):
        self.channel.start_consuming()

    def callback_(self, channel, method, properties, body):
        print("Slave: Received")
        request = json.loads(body)
        eigenspace = self.compute_eigenspace_(request["batch"], request["rank"])
        response = dict()
        response["batchId"] = request["batchId"]
        response["eigenspace"] = eigenspace.to_list()
        self.send_to_master_(str(json.dumps(response)))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_master_(self, message):
        self.channel.basic_publish(exchange='', routing_key='master', body=message)

    def compute_eigenspace_(self, batch, rank):
        sigma = np.zeros((batch[0].shape[0], batch[0].shape[1]))
        for x in batch:
            vec = np.array(x)
            sigma += vec @ vec.T
        sigma = sigma / len(batch)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        return eigenvectors[:, 0:rank]


class MasterNode(Node):
    def __init__(self, broker_host):
        super().__init__(broker_host)
        self.channel.basic_consume(queue='master', on_message_callback=self.callback_)

    def start(self):
        test_data = dict()
        test_data["batchId"] = 0
        test_data["rank"] = 5
        test_data["batch"] = list()
        for i in range(0, 10):
            test_data["batch"].append(np.random.rand(5, 1).tolist())
        self.send_to_slave_(str(json.dumps(test_data)))

        self.channel.start_consuming()

    def callback_(self, channel, method, properties, body):
        print("Master: received")
        request = json.loads(body)
        batch_id = request["batchId"]
        eigenspace = np.array(request["eigenspace"])
        print(batch_id)
        print(eigenspace)
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_slave_(self, message):
        self.channel.basic_publish(exchange='', routing_key='slave', body=message)


def run_master(broker):
    master = MasterNode(broker)
    master.start()


def run_slave(broker):
    slave = SlaveNode(broker)
    slave.start()


def main():
    parser = argparse.ArgumentParser(description="Multinode PCA")
    parser.add_argument("--mode", help="Mode to run script - slave or master")
    parser.add_argument("--broker", help="Message broker IP address")

    args = parser.parse_args()

    if args.broker is None:
        raise RuntimeError("Broker not specified")

    if args.mode == "slave":
        run_slave(args.broker)
    elif args.mode == "master":
        run_master(args.broker)
    else:
        raise RuntimeError("Mode not specified or specified wrong")


if __name__ == "__main__":
    main()
