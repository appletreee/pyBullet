FROM ethansmin/torch-gpu:v0.1
LABEL maintainer="Shih-Min Yang"

# Set working directory
WORKDIR /

# Install packages
ADD requirements.txt /
RUN apt-get update && \
	apt-get install python3-pip && \
	python3 -m pip install --upgrade setuptools pip && \
	python3 -m pip install --no-cache-dir -r requirements.txt


# Find TensorBoard in the port
EXPOSE 6006

COPY run.sh /
RUN chmod +x run.sh
CMD ["/run.sh"]
