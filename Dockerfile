# Based on a CUDA-enabled image (keeps this consistent with later GPU use)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter inside the container must listen on 0.0.0.0, otherwise port mapping
# (e.g. PyCharm's IDE-managed server on a random host port) can't reach it
RUN mkdir -p /root/.jupyter && printf "%s\n" \
    "c.ServerApp.ip = '0.0.0.0'" \
    "c.ServerApp.allow_remote_access = True" \
    > /root/.jupyter/jupyter_server_config.py

# Keep the container running (useful for PyCharm)
CMD ["sleep", "infinity"]
