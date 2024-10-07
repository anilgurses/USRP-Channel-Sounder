FROM ubuntu:20.04

RUN apt-get update

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common 

RUN DEBIAN_FRONTEND=noninteractive apt install -y autoconf automake build-essential ccache cmake cpufrequtils doxygen ethtool \
    g++ git inetutils-tools libboost-all-dev libncurses5 libncurses5-dev libusb-1.0-0 libusb-1.0-0-dev \
    libusb-dev python3-dev python3-pip curl libudev-dev python3-pyqtgraph   python3-mako python3-numpy \
    python3-requests python3-scipy python3-setuptools dpdk python3-ruamel.yaml

RUN rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/EttusResearch/uhd.git
RUN cd uhd/host && mkdir build && cd build &&\
    cmake .. && make -j 12 && make install && ldconfig

RUN uhd_images_downloader

RUN rm -rf /var/lib/apt/lists/*

RUN uhd_images_downloader

RUN mkdir -p /root/dev/

RUN git clone https://github.com/dronekit/dronekit-python
RUN cd dronekit-python && python3 setup.py install && cd ..
RUN python3 -m pip install cython pyyaml scikit-commpy spidev sparkfun-ublox-gps


WORKDIR /root/dev

COPY ./ /root/dev/

RUN echo export PYTHONPATH=/usr/local/local/lib/python3.8/site-packages/ > ~/.bashrc

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"