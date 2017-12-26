FROM continuumio/anaconda3

ENV PYTHONPATH /code

COPY python /code

RUN cd /code && conda env create && /bin/bash -c "cd /code/python && source activate spike-ring && python -m spacy download en"

CMD "source activate spike-ring && python /code/server.py"
