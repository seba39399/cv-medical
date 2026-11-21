# Docker commands

build-image:
	docker build -t cobre-cnn-gpu -f aws/Dockerfile aws/

run-container:
	docker run --rm --gpus all cobre-cnn-gpu python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

run-container-interactive:
	docker run --rm -it --gpus all cobre-cnn-gpu bash