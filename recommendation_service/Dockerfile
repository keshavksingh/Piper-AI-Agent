FROM python:3.13.3

WORKDIR /app

COPY recommendation_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recommendation_service/server.py .
COPY protos/ ./protos/
COPY shared/ ./shared/
COPY certs/ ./certs/
COPY .env .

CMD ["python", "server.py"]
