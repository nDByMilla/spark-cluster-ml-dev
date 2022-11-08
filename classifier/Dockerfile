FROM bde2020/spark-base:3.3.0-hadoop3.3
COPY ../requirements.txt requirements.txt
RUN apk update
RUN apk add make automake gcc g++ subversion python3-dev
RUN /usr/bin/python3 -m pip install --upgrade pip && /usr/bin/python3 -m pip install -r requirements.txt
COPY entrypoint.sh ./entrypoint.sh
COPY preparation.py ./preparation.py
ENV PYARROW_IGNORE_TIMEZONE=1
RUN chmod 750 ./entrypoint.sh
RUN chmod 750 ./preparation.py

ENTRYPOINT ["./entrypoint.sh"]
