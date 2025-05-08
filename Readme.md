## Documentación del Proyecto LLM con RAG - Despliegue en Producción
INDICE
1.Descripción del Proyecto
2.Arquitectura del Sistema
3.Configuración del Entorno
4.Desarrollo del LLM y RAG
5.Despliegue con Docker
6.Configuración de GPU
7.Acceso Público
8.Mantenimiento

-- 1. Descripción del Proyecto ----------------------------------------------
Sistema de inteligencia artificial que combina:

- Modelo LLM (Mistral) para generación de respuestas
- RAG (Retrieval-Augmented Generation) para consulta de documentos
- Interfaz Gradio para interacción con usuarios
- Backend FastAPI para procesamiento de peticiones

-- 2. Arquitectura del Sistema -----------------------------------------------
graph TD
    A[Frontend Gradio] --> B[Backend FastAPI]
    B --> C[Ollama LLM]
    B --> D[Vector Store]
    D --> E[Documentos PDF/PPT/Word]

-- 3. Configuración del Entorno Docker con GPU NVIDIA --------------------------
Requisitos
- Servidor Ubuntu 22.04 LTS
- GPU NVIDIA con drivers 550+
- Docker 24+ y Docker Compose 2.20+
- 16GB RAM mínimo (32GB recomendado)
- 50GB espacio en disco
- Hardware: Servidor con GPU NVIDIA (ej: GTX 1060, RTX 3090, etc.)
- Sistema Operativo: Ubuntu 22.04 LTS (recomendado)
- Conexión a Internet: Para descargar drivers y paquetes
- Acceso root: Permisos de administrador (sudo)
3.1. Identificar tu GPU
comando : lspci | grep -i nvidia
Deberías ver algo como: 01:00.0 VGA compatible controller: NVIDIA Corporation GP106 [GeForce GTX 1060 6GB]

3.2. Instalar Drivers NVIDIA (versión 550)
bash
# Agregar repositorio de drivers
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Instalar drivers recomendados
sudo ubuntu-drivers autoinstall

# O instalar una versión específica (ej: 550)
sudo apt install nvidia-driver-550 -y

# Reiniciar el sistema
sudo reboot

3.3. Verificar Instalación
nvidia-smi
Salida esperada: Tabla con información de tu GPU, versión de driver y procesos activos.
------------------------------------------------------------------------------------------
4. Instalación de Docker y Docker Compose
4.1. Instalar Dependencias
bash
sudo apt update
sudo apt install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    apt-transport-https \
    software-properties-common -y
4.2. Agregar Repositorio Oficial de Docker
bash
# Agregar clave GPG
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Agregar repositorio
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
4.3. Instalar Docker Engine y Docker Compose
bash
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin -y
4.4. Configurar Permisos
bash
sudo usermod -aG docker $USER
newgrp docker  # Actualizar grupo sin cerrar sesión

# Verificar instalación
docker run hello-world
------------------------------------------------------------------------------------------
5. Instalación Manual del NVIDIA Container Toolkit
5.1. Descargar Paquetes Manualmente
bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit-base_1.14.6-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-container-toolkit_1.14.6-1_amd64.deb
5.2. Instalar Dependencias
bash
sudo apt install libnvidia-container1 libnvidia-container-tools -y
5.3. Instalar Paquetes Descargados
bash
sudo dpkg -i nvidia-container-toolkit-base_1.14.6-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit_1.14.6-1_amd64.deb
sudo apt --fix-broken install  # Corregir dependencias si es necesario
5.4. Configurar Docker para Usar GPU
bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
5.5. Verificar Instalación
bash
docker info | grep -i runtime
Salida esperada: Default Runtime: nvidia

------------------------------------------------------------------------------------------

6. Paso 4: Configurar Proyecto con Docker Compose
6.1. Estructura de Archivos
/proyecto
├── docker-compose.yml
├── .env
└── ollama/
    └── models/  # Para persistir modelos descargados
6.2. Configuración de docker-compose.yml
yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ./ollama/models:/root/.ollama  # Persistir modelos
    environment:
      - OLLAMA_NO_CUDA=0  # Forzar uso de GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # Número de GPUs
              capabilities: [gpu, utility]

volumes:
  ollama_data:

-------------------------------------------------------------------------------------------
7. Diagrama de Flujo de la Configuración
graph TD
    A[Host con GPU NVIDIA] --> B[Drivers NVIDIA]
    B --> C[NVIDIA Container Toolkit]
    C --> D[Docker Runtime]
    D --> E[Contenedor con --gpus all]
    E --> F[Acceso a GPU desde el contenedor]


----------------------------------------------------------------------------------------------
8. Desarrollo del LLM y RAG
Estructura del Código
## 1. chat2_llm.py - Núcleo del RAG
## 2. api.py - Backend FastAPI
## 3. frontend_llm_gradio.py - Interfaz de Usuario

------------------------------------------------------------------------------------------------
9. Despliegue con Docker
Estructura de Archivos
/proyecto
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── api.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   └── frontend_llm_gradio.py
└── data/
    ├── pdfs/
    ├── ppts/
    └── words/

docker-compose.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    depends_on: ["ollama"]
    environment:
      - OLLAMA_HOST=http://ollama:11434

  gradio:
    build: ./frontend
    ports: ["7860:7860"]
    depends_on: ["backend"]

-------------------------------------------------------------------------------------------------------
10. Configuración de GPU
Verificación Inicial
bash
nvidia-smi
# Debe mostrar tu GPU y versión de driver

Configuración Específica
Optimizar Ollama para GPU:

bash
docker exec -it ollama ollama pull mistral
Monitorización:

bash
watch -n 1 nvidia-smi

---------------------------------------------------------------------------------------------------------------
11. Actualización del Sistema
bash
# 1. Detener servicios
docker-compose down

# 2. Actualizar imágenes
docker-compose pull

# 3. Reiniciar
docker-compose up -d --build
