FROM ethansmin/torch-gpu:v0.1
LABEL maintainer="Shih-Min Yang"

# Set working directory
WORKDIR /

# Install packages
RUN apt-get update && \
	apt-get install python3-pip && \
	pip3 install --upgrade pip3 && \
	pip3 install -r requirements.txt

# Change authority
RUN chmod +x run.sh

# Sharing volume
VOLUME .

# Find TensorBoard in the port
EXPOSE 6006

COPY run.sh /
CMD ["/run.sh"]
