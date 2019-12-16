FROM rabbitmq:alpine

# Install packages
RUN apk update
RUN apk add gcc gfortran build-base wget freetype-dev libpng-dev openblas-dev
RUN apk add python3 python3-dev py3-pip py3-virtualenv git

# Clone repo
WORKDIR /home/root
RUN git clone "https://github.com/TimeEscaper/distributed_eigenspaces.git"
WORKDIR /home/root/distributed_eigenspaces
RUN git checkout mqtt-setup

# Activate virtualenv
RUN python3 -m virtualenv --python=/usr/bin/python3 venv
RUN . /home/root/distributed_eigenspaces/venv/bin/activate && pip3 install --no-cache-dir -r requirements.txt

