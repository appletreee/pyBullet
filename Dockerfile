FROM ethansmin/torch-gpu:v0.1
LABEL maintainer="Shih-Min Yang"

# Set working directory
WORKDIR /

# Update pip3
RUN pip install --upgrade pip

# Install packages
RUN pip3 install -r requirements.txt

# Change
RUN chmod +x run.sh

# Find TensorBoard in port
EXPOSE 6006

COPY run.sh /
CMD ["/run.sh"]
