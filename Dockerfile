FROM python:3.10-slim
WORKDIR /app

COPY . /app

VOLUME /app/data
RUN pip3 install -r requirements.txt

RUN chmod +x /app/baseline.py
RUN chmod +x /app/make_submission.py

CMD ["python3","/app/make_submission.py"]
