FROM python:3.7

WORKDIR /trader

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "-m", "src.run"]
