FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y     postgresql-client     gcc     python3-dev     libpq-dev     && rm -rf /var/lib/apt/lists/*

COPY requirements_new.txt .
RUN pip install --no-cache-dir -r requirements_new.txt

COPY . .

RUN useradd -m -r healthtracker && chown -R healthtracker:healthtracker /app
USER healthtracker

EXPOSE 5000

ENV FLASK_APP=flask_app.py
ENV FLASK_ENV=production

CMD ["python", "flask_app.py"]