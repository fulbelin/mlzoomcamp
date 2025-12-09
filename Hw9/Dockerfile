FROM agrigorev/model-2025-hairstyle:v1

# Install dependencies
RUN pip install --no-cache-dir \
    pillow \
    numpy \
    onnxruntime \
    torchvision \
    torch

# Copy lambda code
COPY lambda_function.py .

CMD ["lambda_function.lambda_handler"]
