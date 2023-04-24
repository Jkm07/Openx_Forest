from continuumio/anaconda3

COPY env.yaml .

RUN conda env create -f env.yaml

WORKDIR CoverTypePredict

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml_server", "python", "server.py"]