# # A dockerfile must always start by importing the base image.
# # We use the keyword 'FROM' to do that.
# # In our example, we want import the python image.
# # So we write 'python' for the image name and 'latest' for the version.
# FROM python:latest

# RUN pip install torch \
#     && pip install numpy 

# # In order to launch our python code, we must import it into our image.
# # We use the keyword 'COPY' to do that.
# # The first parameter 'main.py' is the name of the file on the host.
# # The second parameter '/' is the path where to put the file on the image.
# # Here we put the file at the image root folder.
# COPY TorchLearn.py /

# # We need to define the command to launch when we are going to run the image.
# # We use the keyword 'CMD' to do that.
# # The following command will execute "python ./main.py".
# CMD [ "python", "./TorchLearn.py" ]


# Use the official PyTorch image with CUDA support. If you don't need CUDA, you can use a CPU-only image.
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime


# Set the working directory in the container
WORKDIR /app

# Copy your script into the container
COPY TorchLearn.py /app/TorchLearn.py

# Install any additional dependencies (numpy is already installed in the base image)
RUN pip install --no-cache-dir numpy

# Run the script
CMD ["python", "/app/TorchLearn.py"]