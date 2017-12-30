import argparse
from functools import reduce
import glob
import json
import math
import os
import os.path as path
import shutil

from funcy import *
import funcy
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter
import torch
import torch.autograd as autograd
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchtext.vocab as vocab

from flask import Flask, jsonify, request
import spacy

from spike_ring.clojure import *

parser = argparse.ArgumentParser()
parser.add_argument("--timestamp")
parser.add_argument("--production")
production = first(parser.parse_known_args()).production
timestamp = first(parser.parse_known_args()).timestamp
resources_path = "../resources"
runs_path = path.join(resources_path, "runs")

if cuda.is_available() or production:
    embedding = vocab.FastText()
else:
    embedding = vocab.GloVe("6B", 50)

vocabulary_size = first(tuple(embedding.vectors.size()))
embedding_vectors = torch.cat(
    [embedding.vectors, init.kaiming_normal(torch.zeros(1, embedding.dim))])


def get_bidirectional_size(n):
    return n * 2


def apply(f, *more):
    return f(*butlast(more), *last(more))


def multiply(*more):
    if len(more) == 1:
        return first(more)
    return multiply(first(more) * second(more), *rest(rest(more)))


def flatten_batches(x):
    return x.contiguous().view(*if_(x.dim() == 2,
                                    [apply(multiply, tuple(x.size()))],
                                    [-1, last(x.size())]))


class Model(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.embedding = nn.Embedding(embedding_vectors.size(0),
                                      embedding_vectors.size(1))
        self.embedding.weight = nn.Parameter(embedding_vectors)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding.dim,
                            m["hidden_size"],
                            m["num_layers"],
                            bidirectional=True)
        self.article = nn.Linear(get_bidirectional_size(m["hidden_size"]), 4)
        self.inflected = nn.Linear(
            get_bidirectional_size(m["hidden_size"]), 2)
    def forward(self, m):
        output, states = self.lstm(self.embedding(m["input"]), m["states"])
        return {"article": flatten_batches(self.article(output)),
                "inflected": flatten_batches(self.inflected(output)),
                "states": states}


def get_state(m):
    return autograd.Variable(init.kaiming_normal(
        get_cuda(torch.zeros(get_bidirectional_size(m["num_layers"]),
                             m["batch_size"],
                             m["hidden_size"]))))


def get_states(m):
    return tuple(repeatedly(partial(get_state, m), 2))


def if_(test, then, else_):
    if test:
        return then
    return else_



def get_hyperparameter_path():
    return "model/hyperparameter.json"
    # return path.join(runs_path, timestamp, "hyperparameter.json")


get_hyperparameter = compose(json.loads,
                             slurp,
                             partial(get_hyperparameter_path))


def get_glob(m):
    return path.join(resources_path,
                     "dataset",
                     m["dataset"],
                     "split",
                     m["split"],
                     "*")


def get_path_batches(m):
    return take(m["batch_size"],
                partition(len(m["training_paths"]) // m["batch_size"],
                          m["training_paths"]))


def flip(f):
    def g(x, *more):
        if empty(more):
            def h(y, *more_):
                apply(f, y, x, more_)
            return h
        return apply(f, first(more), x, rest(more))
    return g


def get(m, k):
    return m[k]


def vector(*more):
    return tuple(more)


get_index = partial(flip(embedding.stoi.get), vocabulary_size)


def get_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def make_get_variable(k):
    return compose(if_(k == "inputs",
                       identity,
                       flatten_batches),
                   torch.t,
                   partial(flip(autograd.Variable), False),
                   get_cuda,
                   torch.LongTensor,
                   vector)


def get_training_variables_(m):
    return drop(m["step_count"],
                take(m["total_step_count"],
                     apply(map,
                           make_get_variable(m["k"]),
                           map(compose(cycle,
                                       partial(partition,
                                               m["step_size"]),
                                       if_(m["k"] == "inputs",
                                           partial(map,
                                                   get_index),
                                           identity),
                                       flatten,
                                       partial(map,
                                               compose(partial(
                                                   flip(get),
                                                   m["k"]),
                                                   json.loads))),
                               m["raw_batches"]))))


def make_get_training_variables(m):
    def get_training_variables(k):
        return get_training_variables_(
            merge(m, {"k": k,
                      "raw_batches":
                          map(partial(map, slurp),
                              get_path_batches(
                                  merge(m,
                                        {"k": k,
                                         "training_paths": glob.glob(get_glob(
                                             set_in(m,
                                                    ["split"],
                                                    "training")))})))}))
    return get_training_variables


get_optimizer = compose(optim.Adam,
                        partial(filter,
                                partial(flip(getattr), "requires_grad")))


def funcall(f, *more):
    if empty(more):
        return f()
    return apply(f, more)


get_loss = nn.CrossEntropyLoss()


def get_non_training_variable_(m):
    return make_get_variable(m["k"])(
        tuple(map(if_(m["k"] == "inputs",
                      get_index,
                      identity),
                  apply(concat,
                        map(compose(partial(flip(get),
                                            m["k"]),
                                    json.loads),
                            m["raw_jsons"])))))


def get_non_training_variable(m):
    return get_non_training_variable_(set_in(m,
                                             ["raw_jsons"],
                                             map(slurp,
                                                 glob.glob(get_glob(m)))))


def get_precision(target, inference):
    return metrics.precision_score(
        target.data.cpu().numpy(),
        torch.topk(inference,
                   1)[1].data.squeeze().cpu().numpy(),
        average="micro")


def make_add_scalars(m):
    def add_scalars(element):
        m["writer"].add_scalars(first(element),
                                second(element),
                                m["step_count"])
        return element
    return add_scalars


def get_checkpoint_path(s):
    return "model/best.pth.tar"
    # return os.path.join(runs_path, timestamp, "checkpoints", s + ".pth.tar")


def log_tensorboard(m):
    writer = SummaryWriter(path.join(runs_path, timestamp, "tensorboard"))
    walk(make_add_scalars(set_in(m, ["writer"], writer)),
         {"loss": {"training": get_in(m, ["training", "loss"]),
                   "validation": get_in(m, ["validation", "loss"])},
          "precision/article":
              {"training": get_precision(get_in(m,
                                                ["training",
                                                 "target",
                                                 "article"]),
                                         get_in(m,
                                                ["training",
                                                 "output",
                                                 "article"])),
               "validation":
                   get_precision(
                       get_non_training_variable(
                           merge(m,
                                 {"k": "articles",
                                  "split": "validation"})),
                       get_in(m,
                              ["validation",
                               "output",
                               "article"]))},
          "precision/inflected":
              {"training": get_precision(get_in(m,
                                                ["training",
                                                 "target",
                                                 "inflected"]),
                                         get_in(m,
                                                ["training",
                                                 "output",
                                                 "inflected"])),
               "validation":
                   get_precision(
                       get_non_training_variable(
                           merge(m,
                                 {"k": "inflecteds",
                                  "split": "validation"})),
                       get_in(m, ["validation",
                                  "output",
                                  "inflected"]))}})
    writer.close()


def mkdirs(path_):
    os.makedirs(path_, exist_ok=True)


make_parent_directories = compose(mkdirs,
                                  os.path.dirname,
                                  os.path.abspath)


def contains(coll, k):
    return k in coll


def select_keys(m, ks):
    return funcy.select_keys(partial(contains, ks), m)


def state_dict(x):
    return x.state_dict()


def log_checkpoint(m):
    make_parent_directories(get_checkpoint_path(""))
    torch.save(merge(select_keys(m, ["states", "step_count"]),
                     walk_values(state_dict,
                                 select_keys(m, ["model", "optimizer"])),
                     {"optimizer": m["optimizer"].state_dict(),
                      "best": get_in(m, ["validation", "loss"])}),
               get_checkpoint_path("recent"))
    if get_in(m, ["validation", "loss"]) < m["best"]:
        if path.exists(get_checkpoint_path("best")):
            shutil.copy2(get_checkpoint_path("best"),
                         get_checkpoint_path("second"))
        shutil.copy2(get_checkpoint_path("recent"),
                     get_checkpoint_path("best"))


def log(m):
    log_tensorboard(m)
    log_checkpoint(m)


def make_run_batch(m):
    def run_batch(reduction, element):
        m["model"].zero_grad()
        training_output = m["model"]({"input": first(element),
                                      "states": walk(compose(funcall,
                                                             partial(
                                                                 flip(getattr),
                                                                 "detach")),
                                                     reduction["states"])})
        training_loss = get_loss(training_output["article"],
                                 second(element)) + get_loss(
            training_output["inflected"], last(element))
        training_loss.backward()
        m["optimizer"].step()
        if reduction["step_count"] % m["validation_frequency"] == 0:
            validation_output = m["model"](
                {"input":
                     get_non_training_variable(merge(m,
                                                     {"k": "inputs",
                                                      "split": "validation"})),
                 "states": get_states(set_in(m, ["batch_size"], 1))})
            validation_loss = (get_loss(
                validation_output["article"],
                get_non_training_variable(
                    merge(m,
                          {"k": "articles",
                           "split": "validation"}))) + get_loss(
                validation_output["inflected"],
                get_non_training_variable(
                    merge(m,
                          {"k": "inflecteds",
                           "split": "validation"})))).data[0]
            log(merge(m,
                      update_in(reduction, ["step_count"], inc),
                      {"training": {"loss": training_loss.data[0],
                                    "output": training_output,
                                    "target": {"article": second(element),
                                               "inflected": last(element)}},
                       "validation": {"loss": validation_loss,
                                      "output": validation_output}}))
        else:
            validation_loss = math.inf
        return merge(training_output,
                     merge_with(min,
                                # TODO add best step_count
                                {"best": validation_loss},
                                update_in(reduction, ["step_count"], inc)))
    return run_batch


def get_batches(m):
    return apply(partial(map, vector),
                 map(make_get_training_variables(m),
                     ["inputs", "articles", "inflecteds"]))


def copyplus(from_, to):
    make_parent_directories(to)
    return shutil.copy2(from_, to)


def load():
    model = get_cuda(Model(get_hyperparameter()))
    optimizer = get_optimizer(model.parameters())
    if os.path.exists(get_checkpoint_path("recent")):
        checkpoint = torch.load(get_checkpoint_path("recent"), map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        checkpoint = {}
    return merge(get_hyperparameter(),
                 {"best": math.inf,
                  "step_count": 0,
                  "states": get_states(get_hyperparameter())},
                 checkpoint,
                 {"model": model,
                  "optimizer": optimizer})


def train():
    loaded = load()
    reduce(make_run_batch(loaded), get_batches(loaded), loaded)


def infer(coll):
    return {"articles": walk(int, tuple(torch.topk(loaded["model"](
        {"input": make_get_variable("inputs")(walk(get_index, coll)),
         "states": get_states(set_in(loaded, ["batch_size"], 1))})["article"],
                                                   1)[
                                            1].data.squeeze().numpy())),
            "inflecteds": walk(int, tuple(torch.topk(loaded["model"](
                {"input": make_get_variable("inputs")(walk(get_index, coll)),
                 "states": get_states(set_in(loaded, ["batch_size"], 1))})[
                                                         "inflected"],
                                                     1)[
                                              1].data.squeeze().numpy()))}


app = Flask(__name__)

nlp = spacy.load("en")


def get_map(token):
    return {"is_sent_start": token.is_sent_start,
            "is_title": token.is_title,
            "is_upper": token.is_upper,
            "lemma_": token.lemma_,
            "lower_": token.lower_,
            "tag_": token.tag_,
            "text": token.text,
            "text_with_ws": token.text_with_ws,
            "whitespace_": token.whitespace_}


@app.route("/", methods=["POST"])
def index():
    print(request.get_json())
    if request.get_json()["action"] == "parse":
        return jsonify(tuple(map(get_map, (nlp(request.get_json()["input"])))))
    return jsonify(infer(request.get_json()["inputs"]))


if __name__ == "__main__":
    loaded = load()
    app.run("0.0.0.0", debug=True)
