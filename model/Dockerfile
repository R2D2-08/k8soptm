FROM python:3.10

WORKDIR /app

RUN pip install --no-cache-dir flask torch cpuonly requests --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 5000

CMD ["python", "server.py"]
