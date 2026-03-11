FROM python:3.9-slim

WORKDIR /app

#ADD script.py /script.py
#ADD requirements.txt /requirements.txt
#RUN pip3 install -r /requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY script.py .
COPY run.py .
COPY ./models ./models

#ENTRYPOINT ["python", "run.py"]

#ENTRYPOINT [ "python3", "/script.py", "-i", "$inputDataset", "-o", "$outputDir" ]
ENTRYPOINT [ "python3", "/app/script.py", "-i", "$inputDataset", "-o", "$outputDir" ]
