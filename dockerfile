FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 git gcc build-essential -y

COPY . ./
RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "./run.py" ]
