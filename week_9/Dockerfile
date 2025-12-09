FROM agrigorev/model-2025-hairstyle:v1 

RUN pip install onnxruntime keras-image-helper==0.0.2

COPY hair_classifier_v1.onnx ./
COPY hair_classifier_v1.onnx.data ./
ENV MODEL_NAME=hair_classifier_v1.onnx

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]