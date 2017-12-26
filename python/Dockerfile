FROM continuumio/anaconda3

ENV PYTHONPATH /code

COPY . /code

RUN cd /code && conda env create && /bin/bash -c "cd /code && source activate spike-ring && python -m spacy download en"

CMD "cd /code && source activate spike-ring && python server.py"
