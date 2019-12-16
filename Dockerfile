FROM rabbitmq:alpine

# Install packages
RUN apk add python3 git

# Clone repo
RUN git clone "https://github.com/TimeEscaper/distributed_eigenspaces.git"
# TODO continue
