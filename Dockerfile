# Bazujemy na obrazie z CUDA (zapewnia spójność z późniejszym GPU)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

# Ustawiamy folder roboczy wewnątrz kontenera
WORKDIR /app

# Kopiujemy plik z wymaganiami i instalujemy je
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter w kontenerze musi słuchać na 0.0.0.0, inaczej mapowanie portów
# (np. losowy port serwera zarządzanego przez PyCharm) nie ma jak go dosięgnąć
RUN mkdir -p /root/.jupyter && printf "%s\n" \
    "c.ServerApp.ip = '0.0.0.0'" \
    "c.ServerApp.allow_remote_access = True" \
    > /root/.jupyter/jupyter_server_config.py

# Pozostawiamy kontener uruchomionym (użyteczne dla PyCharma)
CMD ["sleep", "infinity"]
