FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install tensorflow flask gunicorn google-cloud-storage
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]