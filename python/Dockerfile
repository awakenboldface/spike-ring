FROM continuumio/anaconda3

ENV PYTHONPATH /code

COPY . /code

RUN apt-get update && apt-get install -y build-essential && cd /code && conda env create && /bin/bash -c "cd /code && source activate spike-ring && python -m spacy download en && python test/test_server.py"

EXPOSE 5000

CMD ["/bin/bash", "-c", "cd /code && source activate spike-ring && python spike_ring/server.py"]
