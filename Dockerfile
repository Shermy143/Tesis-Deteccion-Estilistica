FROM python:3

WORKDIR /app

ADD script.py /script.py
ADD requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt

COPY ./models /app/models
COPY run.py .

#ENTRYPOINT ["python", "run.py"]

ENTRYPOINT [ "python3", "/script.py", "-i", "$inputDataset", "-o", "$outputDir" ]
