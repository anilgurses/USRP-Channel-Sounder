# Channel Sounder

This is a real-time channel sounder with different waveform options. It can easily be configured through the config files located under configs directory. 

## Getting Started

There are 2 options to install channel sounder,

### With Docker

```
# Network Host option can be ignored
$ sudo docker build --network=host -t ch-sounder .
$ sudo docker run -dit --network=host --privileged -v /dev:/dev --name sounder ch-sounder
$ sudo docker exec -it sounder bash
$ cd sounder
$ python3 main.py -c ../config/tx_config.yaml 

```

### Without Docker

If you choose not to use Docker, you can do it by installing all the dependencies. 
 
```
$ sudo apt-get update -y 
$ sudo apt-get install libuhd-dev uhd-host
$ uhd_images_downloader
$ python3 -m pip install -r requirements.txt
$ cd sounder
$ python3 main.py -c ../config/tx_config.yaml 

```

Dataset link:  http://datadryad.org/stash/share/IVEdI1Z9X6gYfOtDu3HuZsmoDy1CFhi4AXU2jUo_jsg

## USAGE 

TBD

## Cite 
