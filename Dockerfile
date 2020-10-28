FROM ubuntu:latest
COPY requirements.txt /usr/local/bin/requirements.txt
RUN apt-get update\
	&& apt-get install -y python3-pip python3-dev\
	&& cd /usr/local/bin\
	&& ln -s /usr/bin/python3 python\
	&& pip3 install -r requirements.txt
COPY model.py model.py
COPY cnndata.py cnndata.py
COPY predict.py predict.py
COPY vocab.pkl vocab.pkl
COPY cnn.pt cnn.pt
COPY cnn.txt cnn.txt
COPY own_vec_50.vec own_vec_50.vec
ENTRYPOINT ["python3","predict.py"]
