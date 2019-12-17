#!/usr/bin/env python

import pika
import json
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
        self.channel.basic_consume(queue='hello', on_message_callback=self.callback_)

    def start(self):
        self.channel.start_consuming()

    def callback_(self, channel, method, properties, body):
        request = json.loads(body)
        eigenspace = self.compute_eigenspace_(request["batches"], request["rank"])
        response = dict()
        response["batchId"] = request["batchId"]
        response["eigenspace"] = eigenspace.to_list()
        self.send_to_master_(str(json.dumps(response)))
        channel.basic_ack(delivery_tag=method.delivery_tag)

    def send_to_master_(self, message):
        self.channel.basic_publish(exchange='', routing_key='master', body=message)

    def compute_eigenspace_(self, batches, rank):
        sigma = np.zeros((len(batches[0]), len(batches[0])))
        for x in batches:
            sigma += x @ x.T
        sigma = sigma / len(batches)
        eigenvalues, eigenvectors = np.linalg.eig(sigma)
        return eigenvectors[:, 0:rank]



def run_master():
    pass


def run_slave():
    slave = SlaveNode("localhost")
    slave.start()


def main():
    pass


if __name__ == "__main__":
    main()