FROM nvcr.io/nvidia/pytorch:22.05-py3

WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt