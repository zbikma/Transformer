FROM python:3.8-slim
WORKDIR /app
COPY . /app
# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt
CMD [ "python","matrixmultiply.py" ]