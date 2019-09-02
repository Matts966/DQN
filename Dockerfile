FROM python:3.6.7
COPY dqn.py /
RUN python -m pip install chainer==2.1.0 \
    chainerrl==0.2.0 \
    numpy==1.16.2
ENTRYPOINT python /dqn.py
