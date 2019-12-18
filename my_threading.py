#!/usr/bin/env python

import threading


class Slave(threading.Thread):
    """
    Simple threading class that accept function
    Function(target) execute with it's  arguments(args) to use in this function.
    """
    def __init__(self, target, *args):
        threading.Thread.__init__(self, target=target, args=args)

    def run(self):
        self._target(*self._args)


def some_Func(data, key):
    print("some_Func was called : data=%s; key=%s" % (str(data), str(key)))


def main():
    fn = lambda x: print(x)

    for x in range(2):
        mythread = Slave(fn, 6)
        mythread.start()
        mythread.join()

if __name__ == '__main__':
    main()
