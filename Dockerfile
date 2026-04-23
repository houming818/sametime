FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn9-devel

WORKDIR /src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Add entry point
ENTRYPOINT ["python", "-u", "/src/scripts/eval/run_benchmark.py"]
