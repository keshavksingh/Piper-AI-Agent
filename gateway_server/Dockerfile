FROM python:3.13.3

WORKDIR /app

COPY gateway_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway_server/main.py .
COPY gateway_server/templates ./templates
COPY gateway_server/static ./static
COPY protos/ ./protos/
COPY shared/ ./shared/
COPY certs/ ./certs/
COPY .env .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
